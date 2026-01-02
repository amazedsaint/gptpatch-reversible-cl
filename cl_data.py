import json
from typing import Iterator, Optional, Dict, Any, List, Callable

import torch
from torch.utils.data import IterableDataset, get_worker_info


def iter_text_from_files(
    paths: List[str],
    text_field: Optional[str] = None,
    repeat: bool = False,
) -> Iterator[str]:
    """
    Yield text from .txt (line-based) or .jsonl (one JSON per line).
    If jsonl, reads text_field.
    """
    while True:
        for p in paths:
            if p.endswith(".jsonl"):
                if not text_field:
                    raise ValueError("jsonl requires text_field")
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        txt = obj.get(text_field, "")
                        if txt:
                            yield txt
            else:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield line
        if not repeat:
            return


def iter_text_from_hf(
    dataset: str,
    subset: Optional[str],
    split: str,
    text_field: str,
    streaming: bool = True,
):
    """
    Hugging Face datasets iterator. Requires internet or cached datasets.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset, subset, split=split, streaming=streaming)
    for ex in ds:
        txt = ex.get(text_field, "")
        if txt:
            yield txt


class TokenBlockDataset(IterableDataset):
    """
    Converts a text stream into fixed-length token blocks for causal LM.

    Each yielded item is:
      {"input_ids": LongTensor[block_size], "attention_mask": LongTensor[block_size]}
    """

    def __init__(
        self,
        tokenizer,
        block_size: int,
        eos_token_id: int,
        text_iter: Optional[Iterator[str]] = None,
        text_iter_factory: Optional[Callable[[], Iterator[str]]] = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        if text_iter_factory is None and text_iter is None:
            raise ValueError("Provide either text_iter_factory (preferred) or text_iter")
        self.text_iter = text_iter
        self.text_iter_factory = text_iter_factory
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        self.eos = int(eos_token_id)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        buf: List[int] = []
        text_iter = self.text_iter_factory() if self.text_iter_factory is not None else self.text_iter

        # Shard across distributed ranks and dataloader workers by skipping items.
        info = get_worker_info()
        worker_id = int(info.id) if info is not None else 0
        num_workers = int(info.num_workers) if info is not None else 1
        shard_rank = self.rank * num_workers + worker_id
        shard_world = self.world_size * num_workers

        for i, text in enumerate(text_iter):
            if (i % shard_world) != shard_rank:
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 0:
                continue
            ids.append(self.eos)
            buf.extend(ids)

            while len(buf) >= self.block_size:
                block = buf[: self.block_size]
                buf = buf[self.block_size :]
                input_ids = torch.tensor(block, dtype=torch.long)
                attn = torch.ones_like(input_ids, dtype=torch.long)
                yield {"input_ids": input_ids, "attention_mask": attn}


class ReplayBuffer:
    """
    Stores fixed-length token blocks on CPU. Samples return GPU tensors when requested.
    """

    def __init__(self, max_size: int, block_size: int, seed: int = 0):
        self.max_size = int(max_size)
        self.block_size = int(block_size)
        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(int(seed))
        self.storage = torch.empty((0, self.block_size), dtype=torch.int32)

    def __len__(self):
        return int(self.storage.shape[0])

    @torch.no_grad()
    def add(self, input_ids: torch.Tensor):
        """
        input_ids: LongTensor[B, block_size] on any device.
        """
        x = input_ids.detach().to("cpu", dtype=torch.int32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.shape[1] != self.block_size:
            raise ValueError(f"ReplayBuffer expects block_size={self.block_size}, got {x.shape}")

        if len(self) == 0:
            self.storage = x[: min(x.shape[0], self.max_size)]
            return

        # Reservoir-like: fill first, then random replace
        for row in x:
            if len(self) < self.max_size:
                self.storage = torch.cat([self.storage, row.view(1, -1)], dim=0)
            else:
                j = int(torch.randint(low=0, high=len(self) + 1, size=(1,), generator=self.rng).item())
                if j < self.max_size:
                    self.storage[j] = row

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        if len(self) == 0:
            raise RuntimeError("ReplayBuffer is empty")
        idx = torch.randint(low=0, high=len(self), size=(batch_size,), generator=self.rng)
        x = self.storage[idx].to(device=device, dtype=torch.long)
        attn = torch.ones_like(x, dtype=torch.long)
        labels = x.clone()
        return {"input_ids": x, "attention_mask": attn, "labels": labels}
