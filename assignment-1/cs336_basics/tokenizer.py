from __future__ import annotations

from collections.abc import Iterable, Iterator

import regex as re


# GPT‑2 style pre‑tokenization pattern (same as used in train_bpe.py)
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        # id -> bytes mapping
        self.vocab: dict[int, bytes] = dict(vocab)
        # list of (bytes, bytes) merges in creation order
        self.merges: list[tuple[bytes, bytes]] = list(merges)
        self._merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }

        # bytes -> id mapping for fast lookup during encoding
        self._bytes_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}

        # Normalize and add any missing special tokens to the vocab
        self.special_tokens: list[str] = list(special_tokens or [])
        self._special_token_to_id: dict[str, int] = {}
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            tok_id = self._bytes_to_id.get(token_bytes)
            if tok_id is None:
                tok_id = len(self.vocab)
                self.vocab[tok_id] = token_bytes
                self._bytes_to_id[token_bytes] = tok_id
            self._special_token_to_id[token] = tok_id

        # Compiled regex for splitting on special tokens, if any
        self._special_split_re = None
        if self.special_tokens:
            # Sort by length so longer / overlapping tokens take precedence
            escaped = [re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)]
            pattern = "(" + "|".join(escaped) + ")"
            self._special_split_re = re.compile(pattern)

        # Precompile pre‑tokenization regex
        self._pat_re = re.compile(PAT)

    def _bpe(self, piece_bytes: bytes) -> list[bytes]:
        if not piece_bytes:
            return []

        word: list[bytes] = [bytes([b]) for b in piece_bytes]
        if len(word) == 1:
            return word

        pairs = set(zip(word, word[1:]))
        while True:
            best_pair: tuple[bytes, bytes] | None = None
            best_rank: int | None = None
            for pair in pairs:
                rank = self._merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            first, second = best_pair
            new_word: list[bytes] = []
            i = 0
            while i < len(word):
                if i + 1 < len(word) and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word

            if len(word) == 1:
                break
            pairs = set(zip(word, word[1:]))

        return word

    def _encode_ordinary_text(self, text: str) -> list[int]:
        """Encode text that does not contain special tokens."""
        ids: list[int] = []
        for match in self._pat_re.finditer(text):
            piece = match.group(0)
            if not piece:
                continue
            for tok in self._bpe(piece.encode("utf-8")):
                ids.append(self._bytes_to_id[tok])
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs."""
        if not text:
            return []

        if not self._special_split_re:
            return self._encode_ordinary_text(text)

        ids: list[int] = []
        # Split while keeping the special tokens as separate elements
        parts = self._special_split_re.split(text)
        for part in parts:
            if not part:
                continue
            if part in self._special_token_to_id:
                ids.append(self._special_token_to_id[part])
            else:
                ids.extend(self._encode_ordinary_text(part))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode an iterable of strings into a stream of token IDs."""
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back into a string."""
        if not ids:
            return ""
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")
