"""
Train or load a ByteLevel BPE tokenizer (vocab_size=8192).

Usage:
  from tokenizer import BPETokenizer, train_tokenizer
"""

from __future__ import annotations
import os
import time
import logging
from typing import Iterator

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

logger = logging.getLogger("tokenizer")

SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>"]


def train_tokenizer(
    text_iter: Iterator[str],
    save_path: str,
    vocab_size: int = 8192,
    min_frequency: int = 2,
) -> "BPETokenizer":
    logger.info("Initializing ByteLevel BPE tokenizer (vocab=%d)", vocab_size)
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    t0 = time.time()
    logger.info("Starting tokenizer training...")
    tokenizer.train_from_iterator(text_iter, trainer=trainer)
    elapsed = time.time() - t0
    logger.info("Tokenizer training complete in %.1fs", elapsed)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    tokenizer.save(save_path)
    logger.info("Tokenizer saved → %s  (vocab=%d)", save_path, tokenizer.get_vocab_size())
    return BPETokenizer(save_path)


class BPETokenizer:
    def __init__(self, path: str):
        self._tok = Tokenizer.from_file(path)
        self.eos_id    = self._tok.token_to_id("<|endoftext|>")
        self.pad_id    = self._tok.token_to_id("<|pad|>")
        self.vocab_size = self._tok.get_vocab_size()
        assert self.eos_id is not None, "missing <|endoftext|> token"

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = self._tok.encode(text).ids
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode_batch(self, texts: list[str], add_eos: bool = False) -> list[list[int]]:
        out = [e.ids for e in self._tok.encode_batch(texts)]
        if add_eos:
            out = [ids + [self.eos_id] for ids in out]
        return out

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)
