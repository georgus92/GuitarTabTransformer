from torch.nn.utils.rnn import pad_sequence
import torch

def guitar_tab_collate_fn(batch):
    """
    Custom DataLoader collate_fn for batches coming from a GuitarTabDataset.
    
    Each *item* in `batch` is expected to be a dict with:
        ├─ "cqt"   : torch.Tensor  – shape (n_bins, T)  Constant-Q spectrogram
        ├─ "tokens": torch.Tensor  – 1-D integer sequence of tab tokens
        └─ "chunk_id": str or int  – identifier (helpful for debugging)

    The function pads variable-length spectrograms and token sequences so they
    can be stacked into a single mini-batch tensor.

    Returns
    -------
    dict with keys
        "cqt"            : FloatTensor (B, T_max, n_bins) – zero-padded on time axis
        "tokens"         : LongTensor  (B, L_max)         – padded with 0 (= PAD)
        "attention_mask" : LongTensor  (B, L_max)         – 1 for real tokens, 0 for PAD
        "chunk_ids"      : list        (B,)               – original IDs (no padding needed)
    """
    # ------------------------------------------------------------------ #
    # 1) Sort by descending spectrogram length so the longest example
    #    comes first.  (Makes RNN/Transformer key-padding masks cheaper.)
    # ------------------------------------------------------------------ #
    batch.sort(key=lambda x: x["cqt"].shape[1], reverse=True)

    try:
        # Transpose each CQT from (n_bins, T) → (T, n_bins) so time is dim-0.
        # pad_sequence expects sequences along dim-0 when batch_first=True.
        cqt_seqs   = [item["cqt"].T for item in batch]   # list[(T_i, n_bins)]
        token_seqs = [item["tokens"]  for item in batch] # list[(L_i,)]
        chunk_ids  = [item["chunk_id"] for item in batch]

        # ------------------------------------------------------------------ #
        # 2) Pad across the *time* dimension (dim-0) for CQT and token tensors
        #    - `batch_first=True` ⇒ result shape starts with (B, ...)
        #    - Tokens are padded with 0, which must match PAD in the vocab
        # ------------------------------------------------------------------ #
        cqt_batch   = pad_sequence(cqt_seqs,   batch_first=True)              # (B, T_max, n_bins)
        token_batch = pad_sequence(token_seqs, batch_first=True, padding_value=0)  # (B, L_max)

        # ------------------------------------------------------------------ #
        # 3) Attention mask: 1 where token ≠ PAD (0), else 0
        #    Many Transformer implementations expect int64 masks.
        # ------------------------------------------------------------------ #
        attention_mask = (token_batch != 0).long()                            # (B, L_max)

        # Package everything into a dict consumed by the training loop
        return {
            "cqt":            cqt_batch,
            "tokens":         token_batch,
            "attention_mask": attention_mask,
            "chunk_ids":      chunk_ids
        }

    except Exception as e:
        # Catch-all helps trace cryptic shape/typing bugs during data loading
        raise RuntimeError(f"guitar_tab_collate_fn failed: {e}")
