from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import ClassVar

import regex as re


PAT: ClassVar[str] = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

class Tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
    
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ):
        with open(vocab_filepath, "rb") as f:
            row_vocab = json.load(f)
        vocab = {int(k): bytes(v, "utf-8") for k, v in row_vocab.items()}
        with open(merges_filepath, "rb") as f:
            row_merges = json.load(f)
        merges = [(bytes(a, "utf-8"), bytes(b, "utf-8")) for a, b in row_merges]
        return cls(vocab, merges, special_tokens)

    def _split_on_special(
        self, text: str
    ) -> Iterator[tuple[bool, str]]:
        """
        Yield (is_special, segment) over the input text.
        Special segments are guaranteed to exactly match one of `special_tokens`.
        """
        if not self._special_pattern:
            if text:
                yield (False, text)
            return

        last_idx = 0
        for match in self._special_pattern.finditer(text):
            start, end = match.span()
            if start > last_idx:
                # Non-special prefix
                yield (False, text[last_idx:start])
            # Special token
            yield (True, match.group(0))
            last_idx = end

        if last_idx < len(text):
            yield (False, text[last_idx:])

    def encode(
        self,
        text: str
    ) -> list[int]:
        bpe_tokens = []
        for char in text:
            if char in self.special_tokens:
                bpe_tokens.append(self.special_tokens.index(char))
            else:
                bpe_tokens.append(self.vocab[char])
        return bpe_tokens

    def encode_iterable(
        self, 
        iterable: Iterable[str]
    ) -> Iterator[int]:

    def decode(
        self,
        ids: list[int]
    ) -> str:

