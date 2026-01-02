#!/usr/bin/env python3
import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cl_data import TokenBlockDataset, iter_text_from_hf  # noqa: E402
from cl_patch_gpt2 import load_patch_state_dict, patch_gpt2_lm  # noqa: E402


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    subset: Optional[str]
    split: str
    text_field: str


def load_patch(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "patch" in ckpt:
        return ckpt["patch"]
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")


@torch.no_grad()
def sample_blocks(tokenizer, spec: DatasetSpec, block_size: int, n_blocks: int) -> torch.Tensor:
    text_iter = iter_text_from_hf(
        dataset=spec.dataset,
        subset=spec.subset,
        split=spec.split,
        text_field=spec.text_field,
        streaming=True,
    )
    ds = TokenBlockDataset(
        tokenizer=tokenizer,
        block_size=block_size,
        eos_token_id=int(tokenizer.eos_token_id),
        text_iter=text_iter,
        rank=0,
        world_size=1,
    )
    blocks = []
    it = iter(ds)
    while len(blocks) < n_blocks:
        ex = next(it)
        blocks.append(ex["input_ids"].to(dtype=torch.long))
    return torch.stack(blocks, dim=0)


@torch.no_grad()
def eval_loss_on_blocks(model, blocks: torch.Tensor, batch_size: int) -> float:
    total_loss = 0.0
    total_tokens = 0
    device = next(model.parameters()).device
    blocks = blocks.to(device=device, dtype=torch.long)
    attn = torch.ones_like(blocks, dtype=torch.long)
    labels = blocks

    for i in range(0, blocks.shape[0], batch_size):
        x = blocks[i : i + batch_size]
        m = attn[i : i + batch_size]
        y = labels[i : i + batch_size]
        out = model(input_ids=x, attention_mask=m, labels=y, use_cache=False)
        tokens = int(m.sum().item())
        total_loss += float(out.loss.item()) * tokens
        total_tokens += tokens

    return total_loss / max(1, total_tokens)


def build_model(model_name: str, patch: dict, device: torch.device):
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model = patch_gpt2_lm(model)
    load_patch_state_dict(model, patch)
    model.eval()
    model.to(device)
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-blocks", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)

    parser.add_argument(
        "--wikitext-start",
        type=str,
        default="runs_user/gpt2_100k_hf/ledger/000_wikitext_start.pt",
    )
    parser.add_argument(
        "--wikitext-commit",
        type=str,
        default="runs_user/gpt2_100k_hf/ledger/000_wikitext_commit.pt",
    )
    parser.add_argument(
        "--imdb-start",
        type=str,
        default="runs_user/gpt2_100k_hf/ledger/001_imdb_start.pt",
    )
    parser.add_argument(
        "--imdb-commit",
        type=str,
        default="runs_user/gpt2_100k_hf/ledger/001_imdb_commit.pt",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    from transformers import AutoTokenizer, set_seed

    set_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    wiki = DatasetSpec("wikitext", "wikitext-2-raw-v1", "train", "text")
    imdb = DatasetSpec("imdb", None, "train", "text")

    print(f"[device] {device}")
    print(f"[blocks] n={args.n_blocks} block_size={args.block_size} batch_size={args.batch_size}")

    print("[sample] wikitext blocks...")
    wiki_blocks = sample_blocks(tokenizer, wiki, args.block_size, args.n_blocks)
    print("[sample] imdb blocks...")
    imdb_blocks = sample_blocks(tokenizer, imdb, args.block_size, args.n_blocks)

    patches = {
        "wikitext_start": load_patch(args.wikitext_start),
        "wikitext_commit": load_patch(args.wikitext_commit),
        "imdb_start": load_patch(args.imdb_start),
        "imdb_commit": load_patch(args.imdb_commit),
    }

    def eval_one(name: str, blocks: torch.Tensor) -> float:
        model = build_model(args.model_name, patches[name], device=device)
        loss = eval_loss_on_blocks(model, blocks, batch_size=args.batch_size)
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None
        return loss

    results = {}
    for name in ["wikitext_start", "wikitext_commit", "imdb_start", "imdb_commit"]:
        results[(name, "wikitext")] = eval_one(name, wiki_blocks)
        results[(name, "imdb")] = eval_one(name, imdb_blocks)
        print(f"[eval] {name} wikitext={results[(name,'wikitext')]:.4f} imdb={results[(name,'imdb')]:.4f}")

    print()
    print(
        "plasticity_wikitext (start->commit on wikitext): "
        f"{results[('wikitext_start','wikitext')]:.4f} -> {results[('wikitext_commit','wikitext')]:.4f}"
    )
    print(
        "plasticity_imdb (start->commit on imdb): "
        f"{results[('imdb_start','imdb')]:.4f} -> {results[('imdb_commit','imdb')]:.4f}"
    )
    print(
        "stability_wikitext (wikitext_commit->imdb_commit on wikitext): "
        f"{results[('wikitext_commit','wikitext')]:.4f} -> {results[('imdb_commit','wikitext')]:.4f}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
