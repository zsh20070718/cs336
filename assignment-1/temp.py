import numpy as np
from pathlib import Path
data_dir = Path("data")
train_txt = data_dir / "TinyStoriesV2-GPT4-train.txt"
valid_txt = data_dir / "TinyStoriesV2-GPT4-valid.txt"

print("trainbytesdone")
valid_bytes = valid_txt.read_text(encoding="utf-8").encode("utf-8")
print("validbytesdone")
train_bytes = train_txt.read_text(encoding="utf-8").encode("utf-8")
print("read done")

np.save(data_dir / "tinystories_train_tokens.npy", np.frombuffer(train_bytes, dtype=np.uint8).astype(np.int32))
print("half done")

np.save(data_dir / "tinystories_valid_tokens.npy", np.frombuffer(valid_bytes, dtype=np.uint8).astype(np.int32))
print("saved:",
      data_dir / "tinystories_train_tokens.npy",
      data_dir / "tinystories_valid_tokens.npy")