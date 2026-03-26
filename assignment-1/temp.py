import numpy as np
from pathlib import Path
from cs336_basics.train_bpe import run_train_bpe
from cs336_basics.tokenizer import Tokenizer
data_dir = Path("data")
train_txt = data_dir / "TinyStoriesV2-GPT4-train.txt"
valid_txt = data_dir / "TinyStoriesV2-GPT4-valid.txt"
vocab_size = 10000
special_tokens = ["<|endoftext|>"]
print("Training BPE...")
vocab, merges = run_train_bpe(
    input_path = train_txt,
    vocab_size = vocab_size,
    special_tokens = special_tokens,
)
tokenizer = Tokenizer(vocab, merges, special_tokens = special_tokens)
print("Encoding train...")
train_ids = np.array(tokenizer.encode(train_txt.read_text(encoding="utf-8")), dtype=np.int32)
np.save(data_dir / "tinystories_train_tokens_bpe10k.npy", train_ids)
print("Encoding valid...")
valid_ids = np.array(tokenizer.encode(valid_txt.read_text(encoding="utf-8")), dtype=np.int32)
np.save(data_dir / "tinystories_valid_tokens_bpe10k.npy", valid_ids)
print("Done.")
print("train shape:", train_ids.shape, "valid shape:", valid_ids.shape)

# import numpy as np
# from pathlib import Path
# data_dir = Path("data")
# train_txt = data_dir / "TinyStoriesV2-GPT4-train.txt"
# valid_txt = data_dir / "TinyStoriesV2-GPT4-valid.txt"

# print("trainbytesdone")
# valid_bytes = valid_txt.read_text(encoding="utf-8").encode("utf-8")
# print("validbytesdone")
# train_bytes = train_txt.read_text(encoding="utf-8").encode("utf-8")
# print("read done")

# np.save(data_dir / "tinystories_train_tokens.npy", np.frombuffer(train_bytes, dtype=np.uint8).astype(np.int32))
# print("half done")

# np.save(data_dir / "tinystories_valid_tokens.npy", np.frombuffer(valid_bytes, dtype=np.uint8).astype(np.int32))
# print("saved:",
#       data_dir / "tinystories_train_tokens.npy",
#       data_dir / "tinystories_valid_tokens.npy")