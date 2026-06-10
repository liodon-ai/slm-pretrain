"""
Download, tokenize, and pack training data into binary shards.

Step 1 — train the tokenizer (if not present):
  python prepare_data.py --train_tokenizer

Step 2 — tokenize all data:
  python prepare_data.py

Shards are flat uint16 numpy arrays written to:
  data/<source>/shard_NNNNN.bin

Each shard is ~200 MB (100M tokens × 2 bytes).
The script is resumable: existing shards are skipped.
"""

from __future__ import annotations
import os
import argparse
import numpy as np
from pathlib import Path

from datasets import load_dataset
from tokenizer import BPETokenizer, train_tokenizer
from config import TrainConfig


# ── Dataset registry ─────────────────────────────────────────────────────────
# (hf_name, config_name_or_None, text_column)
SOURCES: dict[str, tuple[str, str | None, str]] = {
    "fineweb_edu":   ("HuggingFaceFW/fineweb-edu",      "sample-100BT", "text"),
    "cosmopedia_v2": ("HuggingFaceTB/smollm-corpus",    "cosmopedia-v2","text"),
    "fineweb_hq":    ("epfml/FineWeb-HQ",               None,           "text"),
    "finemath":      ("HuggingFaceTB/finemath",         "finemath-3plus","text"),
}

SHARD_TOKENS = 100_000_000   # 100M tokens ≈ 200 MB per shard


# ── Tokenizer training ────────────────────────────────────────────────────────

def _sample_docs(n: int) -> "Iterator[str]":
    """Yield n text docs spread proportionally across sources for tokenizer training."""
    cfg = TrainConfig()
    per_source = {k: max(1, int(n * v)) for k, v in cfg.data_mix.items()}
    for name, count in per_source.items():
        hf_name, config, col = SOURCES[name]
        print(f"  [{name}] sampling {count:,} docs for tokenizer training…")
        ds = load_dataset(hf_name, config, split="train", streaming=True, trust_remote_code=True)
        for i, row in enumerate(ds):
            if i >= count:
                break
            yield row[col]


def ensure_tokenizer(path: str, vocab_size: int, n_docs: int) -> BPETokenizer:
    if os.path.exists(path):
        print(f"Tokenizer found at {path}, loading.")
        return BPETokenizer(path)
    print(f"Training tokenizer on ~{n_docs:,} docs…")
    return train_tokenizer(_sample_docs(n_docs), save_path=path, vocab_size=vocab_size)


# ── Data preparation ──────────────────────────────────────────────────────────

def _count_existing(source_dir: Path) -> tuple[int, int]:
    """Return (shard_count, total_tokens_written) for a source directory."""
    shards = sorted(source_dir.glob("shard_*.bin"))
    total  = sum(len(np.fromfile(s, dtype=np.uint16)) for s in shards)
    return len(shards), total


def prepare(cfg: TrainConfig) -> None:
    tok = BPETokenizer(cfg.tokenizer_path)

    for source_name, weight in cfg.data_mix.items():
        target_tokens = int(cfg.total_tokens * weight)
        source_dir    = Path(cfg.data_dir) / source_name
        source_dir.mkdir(parents=True, exist_ok=True)

        n_shards, written = _count_existing(source_dir)
        if written >= target_tokens:
            print(f"[{source_name}] already complete ({written/1e9:.2f}B / {target_tokens/1e9:.2f}B tokens). Skipping.")
            continue

        print(f"\n[{source_name}] target={target_tokens/1e9:.2f}B tokens  already={written/1e9:.2f}B  remaining={( target_tokens-written)/1e9:.2f}B")

        hf_name, config, col = SOURCES[source_name]
        ds = load_dataset(hf_name, config, split="train", streaming=True, trust_remote_code=True)

        buf:        list[int] = []
        shard_idx:  int       = n_shards

        for row in ds:
            if written >= target_tokens:
                break
            ids = tok.encode(row[col], add_eos=True)
            buf.extend(ids)

            while len(buf) >= SHARD_TOKENS:
                _write_shard(buf[:SHARD_TOKENS], source_dir / f"shard_{shard_idx:05d}.bin")
                buf        = buf[SHARD_TOKENS:]
                written   += SHARD_TOKENS
                shard_idx += 1
                print(f"  {written/1e9:.2f}/{target_tokens/1e9:.2f}B tokens", flush=True)
                if written >= target_tokens:
                    break

        # flush remaining tokens into a final partial shard
        if buf and written < target_tokens:
            keep = min(len(buf), target_tokens - written)
            _write_shard(buf[:keep], source_dir / f"shard_{shard_idx:05d}.bin")
            written += keep

        print(f"[{source_name}] done — {written/1e9:.2f}B tokens in {shard_idx + 1} shards")


def _write_shard(tokens: list[int], path: Path) -> None:
    arr = np.array(tokens, dtype=np.uint16)
    arr.tofile(path)
    print(f"  wrote {path.name}  ({len(arr):,} tokens, {arr.nbytes/1024/1024:.0f} MB)")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_tokenizer", action="store_true",
                        help="Force retrain the tokenizer even if it already exists")
    parser.add_argument("--tokenizer",    default=None, help="Override tokenizer path")
    parser.add_argument("--data_dir",     default=None, help="Override data output dir")
    parser.add_argument("--total_tokens", type=int, default=None, help="Override total token budget")
    parser.add_argument("--vocab_size",   type=int, default=8192)
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.tokenizer:    cfg.tokenizer_path = args.tokenizer
    if args.data_dir:     cfg.data_dir       = args.data_dir
    if args.total_tokens: cfg.total_tokens   = args.total_tokens

    if args.train_tokenizer and os.path.exists(cfg.tokenizer_path):
        os.remove(cfg.tokenizer_path)

    ensure_tokenizer(cfg.tokenizer_path, args.vocab_size, cfg.tokenizer_n_docs)
    prepare(cfg)
