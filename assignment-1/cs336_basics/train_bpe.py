import os

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    vocab = {i: bytes([i]) for i in range(256)}
    
    special_token_set = {}
    for sp in special_tokens:
        special_token_set[len(vocab)] = 1
        vocab[len(vocab)] = sp.encode("utf-8")

    with open(input_path, "rb") as f:
        data = f.read()
    
    import regex as re;
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    rawtext = re.findall(PAT, data.decode("utf-8"))
    print("rawtext done")
    rawtext = [[i for i in sp.encode("utf-8")] for sp in rawtext]
    print("rawtext2 done")
    text_freq = {}
    for i in rawtext:
        text_freq[tuple(i)] = text_freq.get(tuple(i), 0) + 1
    print("text_freq done")
    
    def find_best_pair(text_freq: dict[tuple[int, ...], int]) -> tuple[int, int]:
        pair_freq: dict[tuple[int, int], int] = {}
        for token, occ in text_freq.items():
            for j in range(len(token) - 1):
                pr = (token[j], token[j + 1])
                if pr[0] not in special_token_set and pr[1] not in special_token_set:
                    pair_freq[pr] = pair_freq.get(pr, 0) + occ
        return max(pair_freq, key=pair_freq.get)
    
    def merge_pair(
        text_freq: dict[tuple[int, ...], int], 
        pair: tuple[int, int], 
        new_token: int
    ) -> dict[tuple[int, ...], int]:
        new_text_freq = {}
        for i, occ in text_freq.items():
            new_i = []
            j = 0
            while j < len(i):
                if j < len(i) - 1 and i[j] == pair[0] and i[j + 1] == pair[1]:
                    new_i.append(new_token)
                    j += 2
                else:
                    new_i.append(i[j])
                    j += 1
            tu = tuple(new_i)
            if tu not in new_text_freq:
                new_text_freq[tu] = 0
            new_text_freq[tu] += occ
        return new_text_freq

    merges = []
    while len(vocab) < vocab_size:
        print("len(vocab)", len(vocab))
        pr = find_best_pair(text_freq)
        text_freq = merge_pair(text_freq, pr, len(vocab))
        vocab[len(vocab)] = vocab[pr[0]] + vocab[pr[1]]
        merges.append((vocab[pr[0]], vocab[pr[1]]))
    
    return (vocab, merges)
