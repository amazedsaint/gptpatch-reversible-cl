import os
import json
import copy
from typing import Dict, Any, Optional, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator

from cl_patch_gpt2 import patch_gpt2_lm, freeze_all_but_patches, patch_state_dict, load_patch_state_dict
from cl_data import TokenBlockDataset, ReplayBuffer, iter_text_from_files, iter_text_from_hf


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def kl_distill(student_logits, teacher_logits, attention_mask, T: float):
    """
    Token-masked KL(teacher || student) with temperature.
    student_logits, teacher_logits: [B,S,V]
    attention_mask: [B,S]
    """
    T = float(T)
    # Match causal LM loss alignment: predict token i+1 from position i.
    student_logits = student_logits[:, :-1, :]
    teacher_logits = teacher_logits[:, :-1, :]
    attention_mask = attention_mask[:, 1:]

    s_logp = F.log_softmax(student_logits / T, dim=-1)
    t_p = F.softmax(teacher_logits / T, dim=-1)
    kl = F.kl_div(s_logp, t_p, reduction="none").sum(dim=-1)  # [B,S]
    kl = (kl * attention_mask).sum() / attention_mask.sum().clamp_min(1)
    return kl * (T * T)


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device=device) for k, v in batch.items()}


@torch.no_grad()
def eval_lm_loss(model, batches: List[Dict[str, torch.Tensor]], accelerator: Accelerator) -> float:
    model.eval()
    total_loss = torch.tensor(0.0, device=accelerator.device)
    total_tokens = torch.tensor(0.0, device=accelerator.device)
    for batch in batches:
        batch = _to_device(batch, accelerator.device)
        out = model(**batch)
        # Match causal LM loss token-count: labels are shifted by 1 inside the model loss.
        tokens = batch["attention_mask"][:, 1:].sum()
        total_loss += out.loss * tokens
        total_tokens += tokens
    total_loss = accelerator.gather(total_loss).sum()
    total_tokens = accelerator.gather(total_tokens).sum().clamp_min(1)
    return (total_loss / total_tokens).item()


def make_text_iter(phase_cfg: Dict[str, Any]):
    ptype = phase_cfg["type"]
    if ptype == "hf":
        repeat = bool(phase_cfg.get("repeat", False))

        def _once():
            return iter_text_from_hf(
                dataset=phase_cfg["dataset"],
                subset=phase_cfg.get("subset", None),
                split=phase_cfg.get("split", "train"),
                text_field=phase_cfg.get("text_field", "text"),
                streaming=bool(phase_cfg.get("streaming", True)),
            )

        if not repeat:
            return _once()

        def _loop():
            while True:
                yield from _once()

        return _loop()
    if ptype == "files":
        return iter_text_from_files(
            paths=phase_cfg["paths"],
            text_field=phase_cfg.get("text_field", None),
            repeat=bool(phase_cfg.get("repeat", False)),
        )
    raise ValueError(f"Unknown phase type: {ptype}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    grad_accum = int(cfg.get("grad_accum_steps", 1))
    accelerator = Accelerator(
        mixed_precision=("bf16" if cfg.get("bf16", True) else "no"),
        gradient_accumulation_steps=grad_accum,
    )
    is_main = accelerator.is_main_process

    if is_main:
        os.makedirs(cfg["output_dir"], exist_ok=True)
        os.makedirs(os.path.join(cfg["output_dir"], "ledger"), exist_ok=True)
        os.makedirs(os.path.join(cfg["output_dir"], "probes"), exist_ok=True)
    accelerator.wait_for_everyone()

    set_seed(int(cfg.get("seed", 0)))

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from transformers import AutoTokenizer, GPT2LMHeadModel

    model_name = cfg.get("model_name", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Patching + freezing
    sidecar_cfg = cfg.get("sidecar", {})
    model = patch_gpt2_lm(
        model,
        hidden_mult=float(sidecar_cfg.get("hidden_mult", 0.25)),
        init_scale=float(sidecar_cfg.get("init_scale", 0.1)),
        gate_init_bias=float(sidecar_cfg.get("gate_init_bias", -4.0)),
    )
    model = freeze_all_but_patches(model)

    # Coupling invertibility sanity check (CPU).
    if is_main and bool(cfg.get("check_invertibility", True)):
        blk0 = model.transformer.h[0]
        d = int(model.config.n_embd)
        x = torch.randn(2, 8, d)
        y = blk0.sidecar(x)
        x2 = blk0.sidecar.inverse(y)
        inv_err = (x2 - x).abs().max().item()
        print(f"[invertibility] max_abs_err={inv_err:.6e}")

    # Optional regularizers to encourage patch locality/smallness.
    gate_l1_weight = float(cfg.get("gate_l1_weight", 0.0))
    patch_l2_weight = float(cfg.get("patch_l2_weight", 0.0))
    scale_l2_weight = float(cfg.get("scale_l2_weight", 0.0))

    log_every = int(cfg.get("log_every_steps", 50))
    verify_frozen_base = bool(cfg.get("verify_frozen_base", True))
    verify_patch_grads = bool(cfg.get("verify_patch_grads", True))

    if cfg.get("gradient_checkpointing", True):
        # For patch-only training, ensure checkpointing mode supports param grads
        # even when inputs do not require grad.
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        model.config.use_cache = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable patch parameters found; patching failed.")

    lr = float(cfg.get("lr", 2e-4))
    wd = float(cfg.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)

    model, optimizer = accelerator.prepare(model, optimizer)

    unwrapped_model = accelerator.unwrap_model(model)
    if is_main:
        total_params = sum(p.numel() for p in unwrapped_model.parameters())
        trainable = sum(p.numel() for p in unwrapped_model.parameters() if p.requires_grad)
        print(f"[params] trainable={trainable:,} total={total_params:,} ({100.0*trainable/total_params:.4f}%)")

    block_size = int(cfg.get("block_size", 512))
    replay = ReplayBuffer(
        max_size=int(cfg.get("replay_buffer_size", 2048)),
        block_size=block_size,
        seed=int(cfg.get("seed", 0)) + 123,
    )

    teacher = None

    def make_probe_batches(n_batches: int, batch_size: int):
        batches = []
        for _ in range(n_batches):
            b = replay.sample(batch_size=batch_size, device=torch.device("cpu"))
            batches.append(b)
        return batches

    def patch_reg_and_stats() -> Dict[str, torch.Tensor]:
        gates = []
        patch_l2s = []
        scales = []
        log_scales = []

        for blk in getattr(unwrapped_model.transformer, "h", []):
            g = getattr(blk, "_last_gate", None)
            if g is not None:
                gates.append(g.mean())
            p2 = getattr(blk, "_last_patch_l2", None)
            if p2 is not None:
                patch_l2s.append(p2)

            sidecar = getattr(blk, "sidecar", None)
            if sidecar is not None:
                scales.append(sidecar.scale().mean())
                log_scales.append(sidecar.log_scale.mean())

        def _mean(xs: List[torch.Tensor]) -> torch.Tensor:
            if len(xs) == 0:
                return torch.tensor(0.0, device=accelerator.device)
            return torch.stack(xs).mean()

        gate_l1 = _mean(gates)
        patch_l2 = _mean(patch_l2s)
        scale_mean = _mean([t.detach() for t in scales]) if len(scales) > 0 else torch.tensor(0.0, device=accelerator.device)
        log_scale_l2 = _mean([t.pow(2) for t in log_scales]) if len(log_scales) > 0 else torch.tensor(0.0, device=accelerator.device)

        reg = gate_l1_weight * gate_l1 + patch_l2_weight * patch_l2 + scale_l2_weight * log_scale_l2

        # Drop references to the forward graph to avoid holding it across steps.
        for blk in getattr(unwrapped_model.transformer, "h", []):
            if hasattr(blk, "_last_gate"):
                blk._last_gate = None
            if hasattr(blk, "_last_patch_l2"):
                blk._last_patch_l2 = None

        return {
            "reg": reg,
            "gate_l1": gate_l1.detach(),
            "patch_l2": patch_l2.detach(),
            "scale_mean": scale_mean,
        }

    micro_bs = int(cfg.get("micro_batch_size", 4))
    eval_every = int(cfg.get("eval_every_steps", 500))
    replay_bs = int(cfg.get("replay_batch_size", micro_bs))

    w_replay = float(cfg.get("replay_loss_weight", 1.0))
    w_kl = float(cfg.get("distill_kl_weight", 1.0))
    T = float(cfg.get("distill_temperature", 2.0))

    tol_rel = float(cfg.get("rollback_tolerance_rel", 0.02))
    rollback_lr_mult = float(cfg.get("rollback_lr_mult", 0.5))
    rollback_kl_mult = float(cfg.get("rollback_kl_mult", 2.0))
    max_rollbacks_per_phase = int(cfg.get("max_rollbacks_per_phase", 10))

    probe_sets: List[List[Dict[str, torch.Tensor]]] = []

    global_step = 0
    for phase_idx, phase in enumerate(cfg["phases"]):
        phase_name = phase.get("name", f"phase_{phase_idx}")
        if is_main:
            print(f"\n=== Phase {phase_idx}: {phase_name} ===")

        ds = TokenBlockDataset(
            text_iter_factory=lambda p=phase: make_text_iter(p),
            tokenizer=tokenizer,
            block_size=block_size,
            eos_token_id=int(tokenizer.eos_token_id),
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
        )
        dl = DataLoader(ds, batch_size=micro_bs, num_workers=int(cfg.get("num_workers", 2)))
        dl = accelerator.prepare(dl)
        it = iter(dl)

        accelerator.wait_for_everyone()
        if is_main:
            phase_ckpt_path = os.path.join(
                cfg["output_dir"], "ledger", f"{phase_idx:03d}_{phase_name}_start.pt"
            )
            torch.save({"patch": patch_state_dict(accelerator.unwrap_model(model))}, phase_ckpt_path)
        accelerator.wait_for_everyone()

        baseline_probe_losses = []
        if len(probe_sets) > 0:
            for pset in probe_sets:
                loss0 = eval_lm_loss(model, pset, accelerator)
                baseline_probe_losses.append(loss0)

        max_updates = int(cfg.get("max_steps_per_phase", 2000))
        updates_in_phase = 0
        rollbacks_in_phase = 0
        last_good_patch = patch_state_dict(accelerator.unwrap_model(model))
        last_good_metrics = {"probe_losses": baseline_probe_losses}

        model.train()
        while updates_in_phase < max_updates:
            with accelerator.accumulate(model):
                try:
                    batch_new = next(it)
                except StopIteration:
                    it = iter(dl)
                    batch_new = next(it)

                batch_new["labels"] = batch_new["input_ids"].clone()

                out_new = model(**batch_new)
                loss_new = out_new.loss
                loss = loss_new

                if len(replay) > 0:
                    batch_old = replay.sample(batch_size=replay_bs, device=accelerator.device)

                    out_old = model(**batch_old)
                    loss_old = out_old.loss
                    loss = loss + w_replay * loss_old

                    if teacher is not None and w_kl > 0.0:
                        with torch.no_grad():
                            t_out = teacher(
                                input_ids=batch_old["input_ids"],
                                attention_mask=batch_old["attention_mask"],
                            )
                            t_logits = t_out.logits
                        s_logits = out_old.logits
                        loss_kl = kl_distill(
                            student_logits=s_logits,
                            teacher_logits=t_logits,
                            attention_mask=batch_old["attention_mask"],
                            T=T,
                        )
                        loss = loss + w_kl * loss_kl
                    else:
                        loss_kl = torch.tensor(0.0, device=accelerator.device)
                else:
                    loss_old = torch.tensor(0.0, device=accelerator.device)
                    loss_kl = torch.tensor(0.0, device=accelerator.device)

                # Patch/gate regularization + stats (optional)
                stats = patch_reg_and_stats()
                loss = loss + stats["reg"]

                accelerator.backward(loss)

                if verify_frozen_base and global_step == 0:
                    bad = []
                    for name, p in unwrapped_model.named_parameters():
                        if p.requires_grad:
                            continue
                        if p.grad is not None:
                            bad.append(name)
                            if len(bad) >= 5:
                                break
                    if bad:
                        raise RuntimeError(f"Frozen base params got grads (first few): {bad}")

                if verify_patch_grads and global_step == 0:
                    has_patch_grad = False
                    for name, p in unwrapped_model.named_parameters():
                        if not p.requires_grad:
                            continue
                        if p.grad is not None:
                            has_patch_grad = True
                            break
                    if not has_patch_grad:
                        raise RuntimeError(
                            "No gradients found for trainable patch parameters. "
                            "If using gradient checkpointing, ensure it is non-reentrant "
                            "(use_reentrant=False) or disable checkpointing."
                        )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                replay.add(batch_new["input_ids"])

            if accelerator.sync_gradients:
                updates_in_phase += 1
                global_step += 1

                if is_main and log_every > 0 and (global_step % log_every) == 0:
                    msg = (
                        f"[train] step={global_step} "
                        f"loss={loss.detach().item():.4f} "
                        f"new={loss_new.detach().item():.4f} "
                        f"old={loss_old.detach().item():.4f} "
                        f"kl={loss_kl.detach().item():.4f} "
                        f"gate={stats['gate_l1'].item():.4f} "
                        f"patch={stats['patch_l2'].item():.6f} "
                        f"scale={stats['scale_mean'].item():.4f}"
                    )
                    print(msg)

                if (global_step % eval_every) == 0 and len(probe_sets) > 0:
                    model.eval()
                    probe_losses = []
                    for pset in probe_sets:
                        probe_losses.append(eval_lm_loss(model, pset, accelerator))

                    regressed = False
                    for l_now, l0 in zip(probe_losses, baseline_probe_losses):
                        if l0 > 0 and (l_now / l0) > (1.0 + tol_rel):
                            regressed = True
                            break

                    flag = torch.tensor([1 if regressed else 0], device=accelerator.device, dtype=torch.int64)
                    if accelerator.num_processes > 1:
                        torch.distributed.broadcast(flag, src=0)
                    regressed = bool(flag.item())

                    if regressed:
                        load_patch_state_dict(accelerator.unwrap_model(model), last_good_patch)
                        optimizer.state.clear()
                        for pg in optimizer.param_groups:
                            pg["lr"] = float(pg.get("lr", lr)) * rollback_lr_mult
                        w_kl = w_kl * rollback_kl_mult
                        rollbacks_in_phase += 1
                        if is_main:
                            print(f"[rollback] step={global_step} probes={probe_losses} baseline={baseline_probe_losses}")
                            rb_path = os.path.join(
                                cfg["output_dir"],
                                "ledger",
                                f"{phase_idx:03d}_{phase_name}_rollback_step{global_step:09d}.pt",
                            )
                            torch.save(
                                {
                                    "patch": last_good_patch,
                                    "global_step": global_step,
                                    "phase_idx": phase_idx,
                                    "phase_name": phase_name,
                                    "probe_losses": probe_losses,
                                    "baseline_probe_losses": baseline_probe_losses,
                                },
                                rb_path,
                            )
                            print(f"[rollback] saved {rb_path}")
                            print(
                                f"[rollback] rollbacks_in_phase={rollbacks_in_phase} lr_now={optimizer.param_groups[0]['lr']:.6g} w_kl_now={w_kl:.6g}"
                            )
                        model.train()
                        if rollbacks_in_phase >= max_rollbacks_per_phase:
                            if is_main:
                                print(f"[rollback] reached max_rollbacks_per_phase={max_rollbacks_per_phase}; ending phase early")
                            break
                        continue

                    last_good_patch = patch_state_dict(accelerator.unwrap_model(model))
                    last_good_metrics = {"probe_losses": probe_losses}
                    if is_main:
                        print(f"[ok] step={global_step} probes={probe_losses}")
                        good_path = os.path.join(
                            cfg["output_dir"],
                            "ledger",
                            f"{phase_idx:03d}_{phase_name}_good_step{global_step:09d}.pt",
                        )
                        torch.save(
                            {
                                "patch": last_good_patch,
                                "global_step": global_step,
                                "phase_idx": phase_idx,
                                "phase_name": phase_name,
                                "metrics": last_good_metrics,
                            },
                            good_path,
                        )
                        print(f"[ok] saved {good_path}")
                    model.train()

        accelerator.wait_for_everyone()

        if len(replay) > 0:
            model.eval()
            probe = make_probe_batches(
                n_batches=int(cfg.get("probe_batches", 16)),
                batch_size=int(cfg.get("probe_batch_size", micro_bs)),
            )
            probe_sets.append(probe)

            if is_main:
                commit_path = os.path.join(
                    cfg["output_dir"], "ledger", f"{phase_idx:03d}_{phase_name}_commit.pt"
                )
                torch.save(
                    {
                        "patch": patch_state_dict(accelerator.unwrap_model(model)),
                        "global_step": global_step,
                        "phase_idx": phase_idx,
                        "phase_name": phase_name,
                        "metrics": last_good_metrics,
                    },
                    commit_path,
                )
                print(f"[commit] saved {commit_path}")

        accelerator.wait_for_everyone()

        teacher_model = copy.deepcopy(accelerator.unwrap_model(model)).eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        teacher = teacher_model.to(accelerator.device)
        accelerator.wait_for_everyone()

    if is_main:
        print("\nDone.")


if __name__ == "__main__":
    main()
