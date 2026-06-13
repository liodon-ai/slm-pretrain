"""
Dataset that reads from tokenized binary shards and yields (x, y) batches.

Each .bin shard is a flat uint16 array of packed tokens.
The dataset samples shards weighted by the data mix, then draws a
random aligned window of length seq_len+1, splitting it into x / y.
"""

from __future__ import annotations
import os
import random
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

logger = logging.getLogger("data")


class ShardedDataset(IterableDataset):
    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        mix: dict[str, float],
        seed: int = 42,
        split: str = "train",
        val_shards: int = 1,
    ):
        self.seq_len = seq_len
        self.split   = split
        self.seed    = seed

        # Collect (shard_path, weight) pairs
        self.weighted: list[tuple[Path, float]] = []
        sources = [d for d in sorted(Path(data_dir).iterdir()) if d.is_dir()]
        total_sources = len(sources)

        for source_dir in sources:
            name   = source_dir.name
            weight = mix.get(name, 1.0 / total_sources)
            shards = sorted(source_dir.glob("shard_*.bin"))
            if not shards:
                continue
            if split == "val":
                shards = shards[:val_shards]
            else:
                shards = shards[val_shards:]
            for s in shards:
                self.weighted.append((s, weight))

        if not self.weighted:
            raise RuntimeError(
                f"No shards found in {data_dir!r}. "
                "Run prepare_data.py first."
            )

        total_w = sum(w for _, w in self.weighted)
        self._paths   = [p for p, _ in self.weighted]
        self._weights = [w / total_w for _, w in self.weighted]  # normalised

        logger.info("[data] %s: %d shards from %d sources, seed=%d",
                    split, len(self._paths), total_sources, seed)

    # Called per DataLoader worker — each worker gets its own RNG seed
    def _make_rng(self) -> random.Random:
        worker = torch.utils.data.get_worker_info()
        extra  = worker.id if worker is not None else 0
        return random.Random(self.seed + extra)

    def __iter__(self):
        rng = self._make_rng()
        while True:
            path  = rng.choices(self._paths, weights=self._weights, k=1)[0]
            data  = np.memmap(path, dtype=np.uint16, mode="r")
            n     = len(data)
            if n < self.seq_len + 1:
                continue
            start = rng.randint(0, n - self.seq_len - 1)
            chunk = torch.from_numpy(
                data[start : start + self.seq_len + 1].astype(np.int64)
            )
            yield chunk[:-1], chunk[1:]


def make_dataloader(
    data_dir: str,
    seq_len: int,
    batch_size: int,
    mix: dict[str, float],
    seed: int = 42,
    split: str = "train",
    num_workers: int = 4,
) -> DataLoader:
    dataset = ShardedDataset(data_dir, seq_len, mix, seed=seed, split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
    )
