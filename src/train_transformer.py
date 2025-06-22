"""
End-to-end driver script for training a TabTransformerModel on pre-computed
GuitarTab chunks.

Workflow
========
1. CLI args → learning-rate, batch-size, dropout, #epochs, resume checkpoint.
2. Build datasets (train / val) from metadata JSON files.
3. Configure model, optimiser, LR scheduler, and AMP gradient scaler.
4. For each epoch:
     • train_one_epoch()   – single pass over train_loader
     • evaluate()          – run on val_loader (no grads)
     • LR scheduler step   – ReduceLROnPlateau
     • Checkpoint & early-stop logic
5. Plot loss curves at the end of training.

Key files imported
------------------
* precomputed_dataset_loader.PrecomputedGuitarTabDataset
* tab_transformer_model_patched.TabTransformerModel
* tab_transformer_training_amp.train_one_epoch / evaluate
"""

import os
import time
import argparse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler
from time import strftime, localtime

# Local modules
from precomputed_dataset_loader import PrecomputedGuitarTabDataset
from guitar_tab_dataset_loader import GuitarTabTokenizer
from tab_transformer_model import TabTransformerModel
from tab_transformer_training import train_one_epoch, evaluate
from guitar_tab_collate import guitar_tab_collate_fn


# ======================================================================
#                           Main entry point
# ======================================================================
def main():
    # ------------------------------------------------------------ #
    # 1) Parse command-line arguments                              #
    # ------------------------------------------------------------ #
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",          type=float, default=1e-4,  help="Initial learning-rate")
    parser.add_argument("--batch_size",  type=int,   default=4,     help="Mini-batch size")
    parser.add_argument("--dropout",     type=float, default=0.1,   help="Dropout rate in Transformer")
    parser.add_argument("--resume_from", type=str,   default=None,  help="Checkpoint (.pt) to resume")
    parser.add_argument("--epochs",      type=int,   default=20,    help="#epochs to run")
    args = parser.parse_args()

    # Device selection --------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ------------------------------------------------------------ #
    # 2) Paths to dataset manifests & checkpoints                  #
    # ------------------------------------------------------------ #
    data_root  = "SynthTab_PrecomputedChunks_precomputed"
    metadata_train_path = f"{data_root}/metadata_train.json"
    metadata_val_path   = f"{data_root}/metadata_val.json"

    vocab_path     = "guitar_tab_vocab_with_extended_techniques.json"
    checkpoint_dir = "checkpoints"
    best_model_dir = "best_models"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    # ------------------------------------------------------------ #
    # 3) Build tokenizer & model                                   #
    # ------------------------------------------------------------ #
    tokenizer   = GuitarTabTokenizer(vocab_path)
    VOCAB_SIZE  = len(tokenizer.token_to_id)

    model       = TabTransformerModel(
        vocab_size=VOCAB_SIZE,
        dropout=args.dropout
    ).to(device)

    optimizer   = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler      = GradScaler()                     # AMP mixed-precision helper

    # ------------------------------------------------------------ #
    # 4) Resume training (optional)                                #
    # ------------------------------------------------------------ #
    start_epoch   = 0
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    if args.resume_from and os.path.exists(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch    = checkpoint["epoch"] + 1
        train_losses   = checkpoint.get("train_losses", [])
        val_losses     = checkpoint.get("val_losses", [])
        best_val_loss  = checkpoint.get("best_val_loss", float("inf"))
        print(f"🔁 Resumed training from: {args.resume_from} (epoch {start_epoch})")

    # LR scheduler – halve LR if val-loss hasn’t improved for 2 epochs
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2,
        min_lr=1e-6
    )

    # ------------------------------------------------------------ #
    # 5) DataLoaders                                               #
    # ------------------------------------------------------------ #
    train_dataset = PrecomputedGuitarTabDataset(metadata_train_path)
    val_dataset   = PrecomputedGuitarTabDataset(metadata_val_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        collate_fn=guitar_tab_collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        collate_fn=guitar_tab_collate_fn,
        pin_memory=True
    )

    # Early-stopping parameters ----------------------------------------
    EARLY_STOPPING_PATIENCE = 5
    epochs_no_improve       = 0

    # ================================================================= #
    #                       6) Training loop                            #
    # ================================================================= #
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}  (start @ {strftime('%Y-%m-%d %H:%M:%S', localtime())})")

        # -------- Training & validation --------
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler=scaler)
        val_loss   = evaluate(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Step LR scheduler
        scheduler.step(val_loss)

        # Console log
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val   Loss: {val_loss:.4f}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")

        # -------- Checkpointing logic --------
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save({
            "epoch":          epoch,
            "model_state":    model.state_dict(),
            "optimizer_state":optimizer.state_dict(),
            "train_losses":   train_losses,
            "val_losses":     val_losses,
            "best_val_loss":  best_val_loss
        }, checkpoint_path)

        # Save best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(best_model_dir, f"best_model_lr{args.lr}_bs{args.batch_size}_drop{args.dropout}.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved {best_model_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No val-loss improvement for {epochs_no_improve} epoch(s)")

        # Early-stopping condition
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # ================================================================= #
    #                    7) Plot loss curves                             #
    # ================================================================= #
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label="Train")
    plt.plot(range(1, len(val_losses)   + 1), val_losses,   marker='s', label="Val")
    plt.title("Training & Validation Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_path = f"train_loss_curve_lr{args.lr}_bs{args.batch_size}_drop{args.dropout}.png"
    plt.savefig(plot_path)
    plt.show()
    print(f"Loss curve saved to {plot_path}")


# Standard Python module guard
if __name__ == "__main__":
    main()
