import os
import json
import torch
from torch.utils.data import Dataset
from pathlib import Path

class PrecomputedGuitarTabDataset(Dataset):
    """
    A lightweight PyTorch Dataset that loads *pre-computed* chunk files.

    Each chunk was previously saved as a single .pt file containing:
        ├─ "cqt"    : FloatTensor  (n_bins, T) – magnitude CQT spectrogram
        └─ "tokens" : LongTensor   (L,)        – padded sequence of token IDs

    The dataset can be initialised in two ways:

    1) metadata_path = ".../metadata.json"
       A manifest produced by the preprocessing script.  The JSON contains a
       list of dicts, each with an "output_path" pointing to its chunk .pt.

    2) metadata_path = ".../some_directory/"
       Directly point to the folder that *contains* .pt files; the constructor
       will glob for "*.pt" and sort them alphabetically.
    """
    # ------------------------------------------------------------------ #
    # Constructor                                                        #
    # ------------------------------------------------------------------ #
    def __init__(self, metadata_path):
        self.metadata_path = Path(metadata_path)

        # -------------------------------------------------------------- #
        # Case 1 – JSON manifest: read it and build the list of .pt paths
        # -------------------------------------------------------------- #
        if self.metadata_path.suffix == ".json":
            with open(self.metadata_path) as f:
                self.metadata = json.load(f)                               # list[dict]
            self.file_list = [Path(entry["output_path"]) for entry in self.metadata]

        # -------------------------------------------------------------- #
        # Case 2 – Treat metadata_path as a directory of .pt files
        # -------------------------------------------------------------- #
        else:
            self.data_dir  = self.metadata_path
            self.file_list = sorted(self.data_dir.glob("*.pt"))

    # ------------------------------------------------------------------ #
    # PyTorch Dataset API                                                #
    # ------------------------------------------------------------------ #
    def __len__(self):
        """Total number of examples available."""
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Load the idx-th chunk, validate shapes/NaNs, normalise the CQT,
        and return a dict that the collate_fn will batch later."""
        pt_path = self.file_list[idx]
        data    = torch.load(pt_path, weights_only=True)                     # {'cqt': Tensor, 'tokens': Tensor}

        cqt    = data["cqt"]
        tokens = data["tokens"]

        # -------------------- Integrity checks ----------------------- #
        if cqt.dim()    != 2:
            raise ValueError(f"CQT shape invalid: {cqt.shape} in {pt_path}")
        if tokens.dim() != 1:
            raise ValueError(f"Tokens shape invalid: {tokens.shape} in {pt_path}")
        if torch.isnan(cqt).any():
            raise ValueError(f"CQT contains NaNs in {pt_path}")
        if torch.isnan(tokens).any():
            raise ValueError(f"Tokens contain NaNs in {pt_path}")

        # -------------------- Feature normalisation ------------------ #
        # Standard-score: zero-mean, unit-variance (helps model training)
        mean = cqt.mean()
        std  = cqt.std()
        cqt  = (cqt - mean) / (std + 1e-6)                 # avoid div-by-zero

        # Return a sample dict; keys match what the collate_fn expects
        return {
            "cqt":      cqt,                  # FloatTensor (n_bins, T_norm)
            "tokens":   tokens,               # LongTensor  (L,)
            "chunk_id": pt_path.stem          # filename without extension
        }
