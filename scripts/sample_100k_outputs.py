#!/usr/bin/env python3
import argparse
import contextlib
from dataclasses import dataclass
import os
import sys
from typing import Iterable

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cl_patch_gpt2 import load_patch_state_dict, patch_gpt2_lm  # noqa: E402


@dataclass(frozen=True)
class Sample:
    name: str
    prompt: str


def _load_patch_blob(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "patch" in ckpt:
        return ckpt
    return {"patch": ckpt}


@contextlib.contextmanager
def force_gate_bias(model, bias: float):
    saved = []
    for block in model.transformer.h:
        if not hasattr(block, "gate"):
            continue
        w = block.gate.proj.weight.detach().clone()
        b = block.gate.proj.bias.detach().clone()
        saved.append((block, w, b))
        with torch.no_grad():
            block.gate.proj.weight.zero_()
            block.gate.proj.bias.fill_(bias)
    try:
        yield
    finally:
        with torch.no_grad():
            for block, w, b in saved:
                block.gate.proj.weight.copy_(w)
                block.gate.proj.bias.copy_(b)


def _encode(tokenizer, text: str, device: torch.device):
    batch = tokenizer(text, return_tensors="pt")
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def max_abs_logit_diff(model_a, model_b, tokenizer, text: str, device: torch.device) -> float:
    batch = _encode(tokenizer, text, device)
    out_a = model_a(**batch, use_cache=False).logits
    out_b = model_b(**batch, use_cache=False).logits
    return (out_a - out_b).abs().max().item()


@torch.no_grad()
def collect_gate_means(model, tokenizer, text: str, device: torch.device) -> list[float]:
    means: list[tuple[int, float]] = []
    handles = []

    for idx, block in enumerate(model.transformer.h):
        if not hasattr(block, "gate"):
            continue

        def hook(_module, _inp, out, idx=idx):
            means.append((idx, out.detach().float().mean().item()))

        handles.append(block.gate.register_forward_hook(hook))

    batch = _encode(tokenizer, text, device)
    model(**batch, use_cache=False)

    for h in handles:
        h.remove()
    means.sort(key=lambda x: x[0])
    return [m for _, m in means]


@torch.no_grad()
def collect_patch_l2(model, tokenizer, text: str, device: torch.device) -> list[float]:
    stats: list[tuple[int, float]] = []
    handles = []

    for idx, block in enumerate(model.transformer.h):
        if not hasattr(block, "sidecar"):
            continue

        def hook(_module, inp, out, idx=idx):
            h_in = inp[0].detach().float()
            h_out = out.detach().float()
            stats.append((idx, (h_out - h_in).pow(2).mean().item()))

        handles.append(block.sidecar.register_forward_hook(hook))

    batch = _encode(tokenizer, text, device)
    model(**batch, use_cache=False)

    for h in handles:
        h.remove()
    stats.sort(key=lambda x: x[0])
    return [v for _, v in stats]


@torch.no_grad()
def lm_loss(model, tokenizer, text: str, device: torch.device, max_length: int = 256) -> float:
    batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    batch = {k: v.to(device) for k, v in batch.items()}
    batch["labels"] = batch["input_ids"].clone()
    return float(model(**batch, use_cache=False).loss.item())


@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, device: torch.device, max_new_tokens: int) -> str:
    batch = _encode(tokenizer, prompt, device)
    out = model.generate(
        **batch,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def _summarize(values: list[float]) -> str:
    if not values:
        return "n/a"
    t = torch.tensor(values)
    return f"mean={t.mean().item():.4f} min={t.min().item():.4f} max={t.max().item():.4f}"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--patch-ckpt",
        type=str,
        default="runs_user/gpt2_100k_hf/ledger/001_imdb_commit.pt",
        help="Path to a ledger checkpoint containing {'patch': state_dict}.",
    )
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    from transformers import AutoTokenizer, GPT2LMHeadModel, set_seed

    set_seed(0)

    ckpt = _load_patch_blob(args.patch_ckpt)
    patch = ckpt["patch"]
    print(f"[ckpt] path={args.patch_ckpt}")
    if "global_step" in ckpt:
        print(f"[ckpt] global_step={ckpt.get('global_step')} phase={ckpt.get('phase_idx')} name={ckpt.get('phase_name')}")
    print(f"[ckpt] patch_params={len(patch)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Baseline (unpatched) model for identity check
    baseline = GPT2LMHeadModel.from_pretrained(args.model_name).eval()

    # Patched model (loads patch weights). We use the same object for patch-on and patch-off generation.
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    model = patch_gpt2_lm(model)
    load_patch_state_dict(model, patch)
    model.eval()

    identity_text = "The movie was"
    with force_gate_bias(model, bias=-100.0):
        diff = max_abs_logit_diff(baseline, model, tokenizer, identity_text, device=torch.device("cpu"))
    print(f"[check] baseline vs patch_off max_abs_logit_diff={diff:.6e}")

    # Move generation model to device
    model.to(device)

    samples = [
        Sample("generic", "Once upon a time"),
        Sample("news", "In a shocking finding, scientists discovered"),
        Sample("wiki", "In mathematics, a group is"),
        Sample("review_pos", "I loved this movie because"),
        Sample("review_neg", "I hated this movie because"),
        Sample("imdb", "Review: This film was"),
    ]

    print(f"[device] {device}")
    print()

    for s in samples:
        print(f"=== {s.name} ===")

        # Metrics on prompt (gate + patch magnitude)
        gates = collect_gate_means(model, tokenizer, s.prompt, device)
        patch_l2 = collect_patch_l2(model, tokenizer, s.prompt, device)
        scale = torch.tensor([b.sidecar.scale().detach().float().item() for b in model.transformer.h if hasattr(b, 'sidecar')])
        print(f"[gate]  {_summarize(gates)}")
        print(f"[patch] {_summarize(patch_l2)}")
        print(f"[scale] mean={scale.mean().item():.4f} min={scale.min().item():.4f} max={scale.max().item():.4f}")

        # Loss on prompt text (lower is better)
        with force_gate_bias(model, bias=-100.0):
            loss_base = lm_loss(model, tokenizer, s.prompt, device)
            gen_base = generate_text(model, tokenizer, s.prompt, device, args.max_new_tokens)
        loss_patch = lm_loss(model, tokenizer, s.prompt, device)
        gen_patch = generate_text(model, tokenizer, s.prompt, device, args.max_new_tokens)

        print(f"[loss] base={loss_base:.4f} patched={loss_patch:.4f}")
        print("--- base (patch off) ---")
        print(gen_base)
        print("--- patched (as trained) ---")
        print(gen_patch)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
