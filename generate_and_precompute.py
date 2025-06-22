"""
Pre-compute 5-second CQT + token chunks from the SynthTab dataset
and save them (one .pt per chunk) for fast training / evaluation.

High-level flow
---------------
1. Walk the audio root to collect matching FLAC ↔ JAMS pairs.
2. For each pair:
      a) Load & resample the audio (mono, 16 kHz).
      b) Slice it into fixed-size 5 s windows.
      c) Filter out windows with too few notes.
      d) Convert each window to:
            • CQT magnitude (|CQT|)  → float32 tensor
            • Guitar-tab token IDs   → int tensor (padded/truncated)
      e) Save both tensors + metadata to <chunk_id>.pt.
3. Dump a manifest metadata.json for later train/val/test splits.
"""

# -------------------------------------------------------------------- #
#                       Imports & helper utilities                     #
# -------------------------------------------------------------------- #
import os
import json
import librosa
import torch
import torchaudio
import numpy as np
from tqdm import tqdm

from guitar_tab_dataset_loader import (
    parse_note_tab_from_jams,  # JAMS → (string, fret, Δt, effect, onset_ms) list
    GuitarTabTokenizer         # maps tab events ↔ integer IDs
)

# -------------------------------------------------------------------- #
#                           Global configuration                       #
# -------------------------------------------------------------------- #
# Audio / preprocessing ------------------------------------------------
SAMPLE_RATE    = 16_000          # target SR after resampling
CHUNK_DURATION = 5.0             # seconds per training example
MIN_NOTES      = 5               # skip silent windows / rests
RMS_THRESHOLD  = 0.005           # silence gate on RMS energy
MAX_TOKENS     = 512             # fixed length input to Transformer

# CQT parameters -------------------------------------------------------
N_BINS         = 84              # 7 octaves × 12 notes
BINS_PER_OCTAVE= 12
FMIN           = librosa.note_to_hz("C1")  # lowest centre freq

# Paths ----------------------------------------------------------------
root_audio_dir  = "/mnt/scratch/od22jecd/SynthTab"                       
jams_root_dir   = "/mnt/scratch/od22jecd/all_jams_midi_V2_60000_tracks/outall"
output_dir      = "/mnt/scratch/od22jecd/SynthTab_PrecomputedChunks_precomputed/all"
vocab_path      = "guitar_tab_vocab_with_extended_techniques.json"

# Init tokenizer & output folder
tokenizer = GuitarTabTokenizer(vocab_path)
os.makedirs(output_dir, exist_ok=True)

metadata = []   # will collect dicts for metadata.json

# -------------------------------------------------------------------- #
#               Helper: resolve a FLAC track to its JAMS path          #
# -------------------------------------------------------------------- #
def get_jams_path(track):
    """
    Given a leaf folder name `track` (e.g. 'Metallica___Take_1'),
    return the absolute path to its .jams file, handling the occasional
    '[?]' / '?' filename mismatch between audio & annotation folders.
    """
    base_jams_dir = os.path.join(jams_root_dir, track)
    if os.path.exists(base_jams_dir):
        for p in os.listdir(base_jams_dir):
            if p.endswith('.jams'):
                return os.path.join(base_jams_dir, p)

    # Fallback: try sanitised name
    fallback_track = track.replace("[?]", "[_]").replace("?", "_")
    for folder in os.listdir(jams_root_dir):
        if folder == fallback_track:
            base_jams_dir = os.path.join(jams_root_dir, folder)
            for p in os.listdir(base_jams_dir):
                if p.endswith('.jams'):
                    return os.path.join(base_jams_dir, p)
    return None  # not found

# -------------------------------------------------------------------- #
#               STEP 1 – Gather every (flac, jams) tuple               #
# -------------------------------------------------------------------- #
track_pairs = []
for subdir, _, files in os.walk(root_audio_dir):
    for file in files:
        if file.endswith('.flac'):
            guitar_style = os.path.basename(os.path.dirname(subdir))
            track        = os.path.basename(subdir)
            track_name   = f"{track}_{guitar_style}"

            jams_path = get_jams_path(track)
            if jams_path:
                track_pairs.append((os.path.join(subdir, file),
                                    jams_path,
                                    track_name))

print(f"Found {len(track_pairs)} tracks")

# -------------------------------------------------------------------- #
#         STEP 2 – Iterate over tracks; slice into 5-s chunks          #
# -------------------------------------------------------------------- #
for flac_path, jams_path, track_name in tqdm(track_pairs, desc="Processing tracks"):
    # -------------------- Load & preprocess audio -------------------- #
    try:
        waveform, sr = torchaudio.load(flac_path)               # (channels, n_samples)
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        waveform = waveform.mean(dim=0).numpy()                 # mono → 1-D np.array
    except Exception as e:
        print(f"Failed to load {flac_path}: {e}", flush=True)
        continue

    # Parse annotation to a note list once per track
    note_events = parse_note_tab_from_jams(jams_path)

    duration = len(waveform) / SAMPLE_RATE  # total seconds

    # ------------------ Slide a 5-second window over audio ------------ #
    for start in np.arange(0, duration - CHUNK_DURATION + 0.001, CHUNK_DURATION):
        end      = start + CHUNK_DURATION
        chunk_id = f"{track_name}_{start:.2f}"
        out_path = os.path.join(output_dir, f"{chunk_id}.pt")

        # --- Gather notes whose onset lies inside [start, end) -------- #
        start_ms = int(start * 1000)
        end_ms   = int(end   * 1000)
        windowed_notes = [
            (s, f, t, tech)                     # strip off 'onset' for tokenizer
            for (s, f, t, tech, onset) in note_events
            if start_ms <= onset < end_ms
        ]
        if len(windowed_notes) < MIN_NOTES:     # too sparse → skip
            continue

        # --- Extract audio slice + check RMS energy ------------------- #
        start_sample = int(start * SAMPLE_RATE)
        end_sample   = int(end   * SAMPLE_RATE)
        chunk_audio  = waveform[start_sample:end_sample]
        rms = librosa.feature.rms(y=chunk_audio).mean()
        if rms < RMS_THRESHOLD:                 # mostly silence → skip
            continue

        # ------------------ Compute features & tokens ----------------- #
        try:
            # |CQT| magnitude
            cqt = librosa.cqt(chunk_audio,
                              sr=SAMPLE_RATE,
                              n_bins=N_BINS,
                              bins_per_octave=BINS_PER_OCTAVE,
                              fmin=FMIN)
            cqt_tensor = torch.tensor(np.abs(cqt), dtype=torch.float32)  # (n_bins, T)

            # Tokenise note events (→ list[int]) then pad/truncate
            token_ids = tokenizer.encode(windowed_notes)                 # variable-length
            token_ids = token_ids[:MAX_TOKENS]                           # truncate
            token_ids += [tokenizer.token_to_id["PAD"]]*(MAX_TOKENS - len(token_ids))
            token_tensor = torch.tensor(token_ids, dtype=torch.long)     # (MAX_TOKENS,)

            # ------------------ Persist to disk (.pt) ----------------- #
            torch.save({
                "chunk_id": chunk_id,
                "cqt":      cqt_tensor,      # shape (n_bins, T)
                "tokens":   token_tensor     # shape (MAX_TOKENS,)
            }, out_path)

            # Append to manifest
            metadata.append({
                "chunk_id":   chunk_id,
                "track_name": track_name,
                "output_path": out_path,
                "start_time":  round(start, 2),
                "duration":    CHUNK_DURATION
            })

        except Exception as e:
            print(f"Failed to precompute {chunk_id}: {e}", flush=True)

# -------------------------------------------------------------------- #
#                      STEP 3 – Save global manifest                    #
# -------------------------------------------------------------------- #
metadata_path = os.path.join(output_dir, "metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print("Saved metadata.json with", len(metadata), "entries")
print("All chunks processed and precomputed.")
