"""
Create or load a (train / val / test) split for pre-computed SynthTab chunks.

* Input  : metadata.json   – master list produced by preprocessing
* Output : split.json      – dict {train: [ids], val: [...], test: [...]}
          metadata_train.json / metadata_val.json / metadata_test.json
            – filtered manifests for each subset
"""

import os
import json
import random
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------- #
#                           Path configuration                         #
# -------------------------------------------------------------------- #
input_dir        = "SynthTab_PrecomputedChunks_precomputed"

metadata_path    = os.path.join(input_dir, "metadata.json")   # master manifest
split_path       = os.path.join(input_dir, "split.json")      # will store ids only

# Subset-specific metadata files (helpful for separate DataLoaders)
metadata_train_path = os.path.join(input_dir, "metadata_train.json")
metadata_val_path   = os.path.join(input_dir, "metadata_val.json")
metadata_test_path  = os.path.join(input_dir, "metadata_test.json")

# -------------------------------------------------------------------- #
#                       1) Load the full manifest                      #
# -------------------------------------------------------------------- #
with open(metadata_path, "r") as f:
    metadata = json.load(f)                     # list[dict], one per chunk

# Map chunk_id → entry for quick lookup
id_to_entry = {entry["chunk_id"]: entry for entry in metadata}
chunk_ids   = list(id_to_entry.keys())

# -------------------------------------------------------------------- #
#             2) Either load an existing split or create one           #
# -------------------------------------------------------------------- #
if os.path.exists(split_path):
    # Re-use deterministic split to keep experiments comparable
    with open(split_path, "r") as f:
        split = json.load(f)
    print("Loaded split from split.json")

else:
    # Create a new split (80 % train, 10 % val, 10 % test)
    random.seed(42)  # for reproducibility

    # temp_ids = 20 % held-out (to be split equally into val & test)
    train_ids, temp_ids = train_test_split(chunk_ids,
                                           test_size=0.2,
                                           random_state=42)
    # Half of the held-out → val, the other half → test
    val_ids, test_ids = train_test_split(temp_ids,
                                         test_size=0.5,
                                         random_state=42)

    split = {
        "train": train_ids,
        "val":   val_ids,
        "test":  test_ids
    }

    # Persist split for future runs
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)

    print("New split saved to split.json")

# -------------------------------------------------------------------- #
#                3) Write subset-specific metadata files               #
# -------------------------------------------------------------------- #
metadata_train = [id_to_entry[cid] for cid in split["train"]]
metadata_val   = [id_to_entry[cid] for cid in split["val"]]
metadata_test  = [id_to_entry[cid] for cid in split["test"]]

with open(metadata_train_path, "w") as f:
    json.dump(metadata_train, f, indent=2)

with open(metadata_val_path, "w") as f:
    json.dump(metadata_val, f, indent=2)

with open(metadata_test_path, "w") as f:
    json.dump(metadata_test, f, indent=2)

print(f"Saved: {len(split['train'])} train, {len(split['val'])} val, {len(split['test'])} test samples")
