"""
train_one_epoch / evaluate
--------------------------
Utility functions for a Transformer-based guitar-tab model.

Key features
------------
* Robust NaN / Inf detection for audio, tokens, model outputs, and loss.
* Automatic Mixed Precision (AMP) support via an optional `scaler`.
* Gradient‐clipping (max-norm = 1.0) to stabilise training.
* Normalisation of CQT input (per-example z-score).
"""

import torch
import torch.nn.functional as F

# ================================================================
#                       TRAINING LOOP
# ================================================================
def train_one_epoch(model, dataloader, optimizer, device, scaler=None):
    """
    One full pass over the training set.

    Parameters
    ----------
    model : nn.Module
        Transformer-based seq-to-seq model.
    dataloader : DataLoader
        Yields batched dicts with keys "cqt", "tokens", "attention_mask".
    optimizer : torch.optim.Optimizer
        Optimiser instance (e.g. AdamW).
    device : torch.device
        CUDA or CPU device to run on.
    scaler : torch.cuda.amp.GradScaler or None
        If provided, enables mixed-precision training (AMP).

    Returns
    -------
    float
        Mean cross-entropy over all *valid* batches (NaN batches skipped).
    """
    model.train()
    total_loss   = 0.0
    valid_batches = 0  # how many batches actually contributed to grad update

    for batch in dataloader:
        # ----------------------------------------------
        # Move tensors to device and unpack batch
        # ----------------------------------------------
        cqt            = batch["cqt"].to(device)          # (B, 84, T)
        tokens         = batch["tokens"].to(device)       # (B, L)
        attention_mask = batch["attention_mask"].to(device)
        chunk_ids      = batch.get("chunk_ids", ["unknown"] * cqt.size(0))

        # ----------------------------------------------
        # Defensive checks on CQT & token tensors
        # ----------------------------------------------
        if torch.isnan(cqt).any() or torch.isinf(cqt).any():
            print(f"Invalid CQT in batch from: {chunk_ids}")
            continue  # skip corrupt batch

        # Normalise CQT
        cqt = torch.nan_to_num(cqt, nan=0.0, posinf=10.0, neginf=-10.0)
        cqt = torch.clamp(cqt, min=-10.0, max=10.0)
        cqt = (cqt - cqt.mean(dim=(1, 2), keepdim=True)) / (
            cqt.std(dim=(1, 2), keepdim=True) + 1e-6
        )

        if torch.isnan(tokens).any() or torch.isinf(tokens).any():
            print(f"Invalid tokens in batch from: {chunk_ids}")
            continue

        # ----------------------------------------------
        # Shift tokens for teacher-forcing
        #   decoder_input  → BOS ... N-1
        #   target         →     ... N
        # ----------------------------------------------
        decoder_input        = tokens[:, :-1]
        target               = tokens[:, 1:]
        decoder_padding_mask = attention_mask[:, :-1].bool()

        # Skip if entire target is PAD
        if (target != 0).sum() == 0:
            print(f"Skipping PAD-only batch from: {chunk_ids}")
            continue

        optimizer.zero_grad()

        # ----------------------------------------------------------
        # Forward + loss (optionally under AMP autocast)
        # ----------------------------------------------------------
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(cqt, decoder_input)  # (B, L-1, vocab)

            if torch.isnan(outputs).any():
                print(f"NaN in model outputs during training from: {chunk_ids}")
                with open("train_nan_issues.log", "a") as f:
                    f.write(f"[Training] NaN in outputs from: {chunk_ids}\n")
                continue

            # Flatten (B*(L-1), vocab) vs (B*(L-1))
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                target.reshape(-1),
                ignore_index=0  # PAD index
            )

            if torch.isnan(loss):
                print(f"NaN loss detected during training from: {chunk_ids}")
                continue

        # ----------------------------------------------------------
        # Back-prop and optimiser step (AMP or FP32)
        # ----------------------------------------------------------
        if scaler:  # mixed precision branch
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:       # pure FP32 branch (with anomaly detection for debug)
            with torch.autograd.detect_anomaly():
                loss.backward()
            optimizer.step()

        # Extra safety clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        total_loss    += loss.item()
        valid_batches += 1

    # Mean loss over non-skipped batches, or NaN if none were valid
    return total_loss / valid_batches if valid_batches > 0 else float("nan")


# ================================================================
#                       EVALUATION LOOP
# ================================================================
def evaluate(model, dataloader, device, scaler=None):
    """
    Validation / test loop (no gradient updates).

    Returns
    -------
    float
        Mean cross-entropy over all *valid* batches.
    """
    model.eval()
    total_loss   = 0.0
    valid_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            cqt            = batch["cqt"].to(device)
            tokens         = batch["tokens"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            chunk_ids      = batch.get("chunk_ids", ["unknown"] * cqt.size(0))

            # ----------- Sanity checks identical to training -----------
            if torch.isnan(cqt).any() or torch.isinf(cqt).any():
                print(f"Invalid CQT in batch from: {chunk_ids}")
                continue

            cqt = torch.nan_to_num(cqt, nan=0.0, posinf=10.0, neginf=-10.0)
            cqt = torch.clamp(cqt, min=-10.0, max=10.0)
            cqt = (cqt - cqt.mean(dim=(1, 2), keepdim=True)) / (
                cqt.std(dim=(1, 2), keepdim=True) + 1e-6
            )

            if torch.isnan(tokens).any() or torch.isinf(tokens).any():
                print(f"Invalid tokens in batch from: {chunk_ids}")
                continue

            decoder_input        = tokens[:, :-1]
            target               = tokens[:, 1:]
            decoder_padding_mask = attention_mask[:, :-1].bool()

            if (target != 0).sum() == 0:
                print(f"Skipping PAD-only batch from: {chunk_ids}")
                continue

            # ----------- Forward pass (AMP allowed) --------------------
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model(cqt, decoder_input)

                if torch.isnan(outputs).any():
                    print(f"NaN in model outputs during eval from: {chunk_ids}")
                    with open("eval_nan_issues.log", "a") as f:
                        f.write(f"[Eval] NaN in outputs from: {chunk_ids}\n")
                    continue

                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    target.reshape(-1),
                    ignore_index=0
                )

                if torch.isnan(loss):
                    print(f"NaN loss detected during eval from: {chunk_ids}")
                    continue

            total_loss    += loss.item()
            valid_batches += 1

    return total_loss / valid_batches if valid_batches > 0 else float("nan")
