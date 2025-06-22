"""
Batch-evaluation of multiple TabTransformer checkpoints on the pre-computed
SynthTab *test* split.  For each checkpoint we compute:

    * Average cross-entropy loss
    * Token-level accuracy (ignoring PAD)
    * String/fret error counts (diagnostic)
    * Technique-specific precision / recall / F1
    * Confusion-matrix heat-map for effects / techniques
    * Pretty-printed prediction vs ground-truth tab for manual inspection

Outputs
=======
    OUTPUT_DIR/
        ├─ tab_predictions_<model_tag>.txt         # human-readable tab strings
        ├─ confusion_matrix_<model_tag>.png        # heat-map
        ├─ metrics_per_class_<model_tag>.csv       # P/R/F1 per technique
        └─ evaluation_summary.csv                  # headline numbers
"""

# --------------------------------------------------------------------- #
#                               Imports                                 #
# --------------------------------------------------------------------- #
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support
)

# Project-local modules
from guitar_tab_dataset_loader import GuitarTabTokenizer
from tab_transformer_model  import TabTransformerModel
from precomputed_dataset_loader    import PrecomputedGuitarTabDataset
from guitar_tab_collate            import guitar_tab_collate_fn


# --------------------------------------------------------------------- #
#                          Global configuration                         #
# --------------------------------------------------------------------- #
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE    = 4
NUM_WORKERS   = 4

VOCAB_PATH    = "guitar_tab_vocab_with_extended_techniques.json"
METADATA_PATH = ("SynthTab_PrecomputedChunks_precomputed/metadata_test.json")

CHECKPOINT_DIR = "best_models"
OUTPUT_DIR     = "batch_test_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHECKPOINTS = sorted(
    [
        os.path.basename(p)                 # keep just the filename; path is added later
        for p in glob(os.path.join(CHECKPOINT_DIR, "best_model_*.pt"))
    ]
)

print(f"Found {len(CHECKPOINTS)} best-model checkpoints:\n  " + "\n  ".join(CHECKPOINTS))


# --------------------------------------------------------------------- #
#                    1) Load tokenizer & DataLoader                     #
# --------------------------------------------------------------------- #
tokenizer   = GuitarTabTokenizer(VOCAB_PATH)
VOCAB_SIZE  = len(tokenizer.token_to_id)

dataset = PrecomputedGuitarTabDataset(METADATA_PATH)
loader  = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=guitar_tab_collate_fn,
    pin_memory=True,
)

# --------------------------------------------------------------------- #
#                         Helper: tab pretty-print                      #
# --------------------------------------------------------------------- #
def format_as_tab(events):
    """
    Render decoded (string, fret, shift_ms, effect) tuples into a
    monospaced ASCII guitar-tab block.
    """
    tab_lines = ["e|", "B|", "G|", "D|", "A|", "E|"]

    i = 0
    while i < len(events):
        group   = []            # notes that sound simultaneously (chord)
        spacing = events[i][2]  # silence before the chord (ms)

        # Collect notes until next inter-onset gap > 0
        while i < len(events) and (len(group) == 0 or events[i][2] == 0):
            group.append(events[i])
            i += 1

        # Represent silence as runs of "---"
        dash_count = max(spacing // 120, 1)
        for s in range(6):
            tab_lines[s] += "---" * dash_count

        # Build fret symbols for this “column”
        column = ["---"] * 6
        for string, fret, _, _ in group:
            line_idx = string - 1           # 1 = high-e → index 0
            fret_str = str(fret)
            # Keep spacing for multi-digit frets
            column[line_idx] = (
                f"-{fret_str}-" if len(fret_str) == 1
                else f"{fret_str}-" if len(fret_str) == 2
                else fret_str
            )

        # Append column to each string line
        for s in range(6):
            tab_lines[s] += column[s]

    return "\n".join(tab_lines)


# --------------------------------------------------------------------- #
#                          2) Evaluation loop                           #
# --------------------------------------------------------------------- #
results = []  # will accumulate (model_tag, loss, acc, macro-P, R, F1)

for ckpt_file in CHECKPOINTS:
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_file)
    if not os.path.exists(ckpt_path):
        print(f"Missing: {ckpt_path}")
        continue

    print(f"\nEvaluating: {ckpt_file}")
    # ---------------- Load model checkpoint ---------------- #
    model = TabTransformerModel(vocab_size=VOCAB_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    # Loss fn - ignore PAD tokens (id = 0)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id["PAD"])

    # Running tallies
    total_loss   = 0.0
    total_tokens = 0
    correct      = 0

    string_errors      = defaultdict(int)      # per-string mismatch count
    all_true_techs     = []                    # flattened lists for metrics
    all_pred_techs     = []
    technique_labels   = sorted(tokenizer.effects)

    # Filenames for this model’s artefacts
    model_tag = ckpt_file.replace(".pt", "")
    log_file  = os.path.join(OUTPUT_DIR, f"tab_predictions_{model_tag}.txt")
    cm_file   = os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_tag}.png")

    # ---------------- Inference over entire test set ---------------- #
    with open(log_file, "w") as log, torch.no_grad():
        for batch in loader:
            cqt           = batch["cqt"].to(DEVICE)
            tokens        = batch["tokens"].to(DEVICE)
            track_ids     = batch["chunk_ids"]

            # Teacher-forcing shift
            input_tokens  = tokens[:, :-1]
            target_tokens = tokens[:, 1:]

            # Forward pass
            logits        = model(cqt, input_tokens)

            # ------- Loss & token-accuracy (ignore PAD) ------------ #
            loss          = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_tokens.reshape(-1),
            )
            total_loss   += loss.item()

            preds        = logits.argmax(dim=-1)

            # mask PAD positions before accuracy
            correct     += (
                (preds == target_tokens)
                .masked_fill(target_tokens == tokenizer.token_to_id["PAD"], False)
                .sum()
                .item()
            )
            total_tokens += (target_tokens != tokenizer.token_to_id["PAD"]).sum().item()

            # -------------- Per-sample analyses -------------------- #
            for i in range(tokens.size(0)):
                true_seq = tokenizer.decode(tokens[i].tolist())
                pred_seq = tokenizer.decode(preds[i].tolist())

                # String-/fret-level mismatch diagnostic
                for (sT, fT, _, _), (sP, fP, _, _) in zip(true_seq, pred_seq):
                    if sT != sP or fT != fP:
                        string_errors[sT] += 1

                # Collect technique labels for confusion matrix
                for (_, _, _, tT), (_, _, _, tP) in zip(true_seq, pred_seq):
                    if tT in technique_labels and tP in technique_labels:
                        all_true_techs.append(tT)
                        all_pred_techs.append(tP)

                # Write pretty tab to log file
                log.write(f"{track_ids[i]}\n")
                log.write("Prediction:\n"   + format_as_tab(pred_seq) + "\n")
                log.write("Ground Truth:\n" + format_as_tab(true_seq) + "\n")
                log.write("=" * 60 + "\n\n")

    # ---------------------------------------------------------------- #
    #                 3) Metrics & visualisations                      #
    # ---------------------------------------------------------------- #
    acc        = 100 * correct / total_tokens
    final_loss = total_loss / len(loader)
    print(f"{model_tag} — Test Loss: {final_loss:.4f} | Accuracy: {acc:.2f}%")

    # ---- Confusion-matrix for playing techniques ----
    cm = confusion_matrix(all_true_techs, all_pred_techs, labels=technique_labels)
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=technique_labels, yticklabels=technique_labels
    )
    plt.title(f"Confusion Matrix — {model_tag}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(cm_file)
    plt.close()

    # ---- Precision / recall / F1 per technique ----
    precision, recall, f1, support = precision_recall_fscore_support(
        all_true_techs,
        all_pred_techs,
        labels=technique_labels,
        zero_division=0,   # avoid NaNs for unseen classes
    )
    per_class_df = pd.DataFrame({
        "technique": technique_labels,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "support":   support,
    })
    per_class_path = os.path.join(OUTPUT_DIR, f"metrics_per_class_{model_tag}.csv")
    per_class_df.to_csv(per_class_path, index=False)
    print(f"Per-class metrics written to {per_class_path}")

    # ---- Macro averages (headline) ----
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_true_techs,
        all_pred_techs,
        average="macro",
        zero_division=0,
    )
    print(
        f"Macro P/R/F1 for {model_tag}: "
        f"{macro_p:.3f} / {macro_r:.3f} / {macro_f1:.3f}"
    )

    # Record for global summary
    results.append((model_tag, final_loss, acc, macro_p, macro_r, macro_f1))


# --------------------------------------------------------------------- #
#                     4) Write overall summary CSV                      #
# --------------------------------------------------------------------- #
summary_path = os.path.join(OUTPUT_DIR, "evaluation_summary.csv")
with open(summary_path, "w") as f:
    f.write("model_tag,test_loss,token_accuracy,macro_precision,macro_recall,macro_f1\n")
    for row in results:
        f.write(
            f"{row[0]},{row[1]:.4f},{row[2]:.2f},"
            f"{row[3]:.3f},{row[4]:.3f},{row[5]:.3f}\n"
        )

print(f"Summary saved to {summary_path}")
