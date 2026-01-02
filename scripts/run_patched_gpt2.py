#!/usr/bin/env python3
import argparse
import contextlib
import os
import sys
from typing import Iterable, Optional

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cl_patch_gpt2 import load_patch_state_dict, patch_gpt2_lm  # noqa: E402


def _load_patch_blob(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "patch" in ckpt:
        return ckpt
    return {"patch": ckpt}


@contextlib.contextmanager
def force_gate_mode(model, gate_mode: str):
    if gate_mode == "trained":
        yield
        return

    if gate_mode not in {"off", "on"}:
        raise ValueError(f"Unknown gate_mode: {gate_mode}")

    bias = -100.0 if gate_mode == "off" else 100.0

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


def _prompts_from_file(path: str) -> list[str]:
    prompts: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


@torch.no_grad()
def generate(model, tokenizer, prompt: str, gen_kwargs: dict) -> str:
    batch = tokenizer(prompt, return_tensors="pt")
    batch = {k: v.to(model.device) for k, v in batch.items()}
    out = model.generate(
        **batch,
        max_new_tokens=int(gen_kwargs["max_new_tokens"]),
        do_sample=bool(gen_kwargs["do_sample"]),
        temperature=float(gen_kwargs["temperature"]),
        top_p=float(gen_kwargs["top_p"]),
        top_k=int(gen_kwargs["top_k"]),
        repetition_penalty=float(gen_kwargs["repetition_penalty"]),
        no_repeat_ngram_size=int(gen_kwargs["no_repeat_ngram_size"]),
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--patch-ckpt",
        type=str,
        default="runs_user/gpt2_100k_hf/ledger/001_imdb_commit.pt",
        help="Path to a ledger checkpoint containing {'patch': state_dict}.",
    )
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--gate",
        type=str,
        default="trained",
        choices=["trained", "off", "on"],
        help="Gate mode: trained (default), off (disable patch), on (force patch on).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=120)

    # Default to sampling to reduce repetition.
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding (can be repetitive).")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.10)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)

    parser.add_argument("--prompt", action="append", default=[], help="Prompt text. May be passed multiple times.")
    parser.add_argument("--prompts-file", type=str, default=None, help="Text file with one prompt per line.")
    parser.add_argument("--interactive", action="store_true", help="Interactive REPL mode.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if not os.path.exists(args.patch_ckpt):
        raise FileNotFoundError(
            f"Patch checkpoint not found: {args.patch_ckpt}. "
            "Training outputs are gitignored; pass --patch-ckpt to a local ledger file."
        )

    from transformers import AutoTokenizer, GPT2LMHeadModel, set_seed

    set_seed(int(args.seed))

    ckpt = _load_patch_blob(args.patch_ckpt)
    patch = ckpt["patch"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    model = patch_gpt2_lm(model)
    load_patch_state_dict(model, patch)
    model.eval()
    model.to(device)

    do_sample = not bool(args.greedy)
    gen_kwargs = {
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": bool(do_sample),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "repetition_penalty": float(args.repetition_penalty),
        "no_repeat_ngram_size": int(args.no_repeat_ngram_size),
    }

    print(f"[ckpt] {args.patch_ckpt}")
    if "global_step" in ckpt:
        print(
            f"[ckpt] global_step={ckpt.get('global_step')} "
            f"phase={ckpt.get('phase_idx')} name={ckpt.get('phase_name')}"
        )
    print(f"[ckpt] patch_params={len(patch)}")
    print(f"[device] {device}")
    print(
        "[decode] "
        f"do_sample={do_sample} "
        f"temperature={gen_kwargs['temperature']:.3f} "
        f"top_p={gen_kwargs['top_p']:.3f} "
        f"top_k={gen_kwargs['top_k']} "
        f"repetition_penalty={gen_kwargs['repetition_penalty']:.3f} "
        f"no_repeat_ngram_size={gen_kwargs['no_repeat_ngram_size']}"
    )
    print(f"[gate] {args.gate}")
    print()

    prompts: list[str] = []
    if args.prompts_file:
        prompts.extend(_prompts_from_file(args.prompts_file))
    prompts.extend([p for p in args.prompt if p and p.strip()])

    if args.interactive:
        print("Enter a prompt. Type 'quit' or 'exit' to stop.")
        with force_gate_mode(model, args.gate):
            while True:
                try:
                    prompt = input("> ").strip()
                except EOFError:
                    print()
                    break
                if not prompt:
                    continue
                if prompt.lower() in {"quit", "exit"}:
                    break
                print(generate(model, tokenizer, prompt, gen_kwargs))
                print()
        return 0

    if not prompts:
        raise ValueError("No prompts provided. Use --prompt, --prompts-file, or --interactive.")

    with force_gate_mode(model, args.gate):
        for i, prompt in enumerate(prompts):
            print(f"=== prompt_{i} ===")
            print(prompt)
            print("--- completion ---")
            print(generate(model, tokenizer, prompt, gen_kwargs))
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

