# Transformer Tablature: Deep Learning for Guitar Transcription

This repository contains all Python modules, training scripts, and utilities needed to reproduce the experiments described in the dissertation. 

---

## 📝 Project summary

The goal is to translate polyphonic guitar audio into human‑readable tablature with expressive technique annotations.  
Key ingredients:

* **Dataset** – [SynthTab](https://github.com/yongyizang/SynthTab) (subset used for storage constraints)  
* **Input** – Constant‑Q Transform (CQT) spectrograms (84 bins, 512‑hop, 16 kHz)  
* **Model** – Custom Transformer encoder‑decoder (≈ 9 M params) implemented in PyTorch  
* **Output tokens** – `(string, fret, duration, technique)` tuples covering bends, slides, hammer‑ons, etc.  

---

## 📁 Repository structure

```
.
├── src/                         
│   ├── guitar_tab_dataset_loader.py
│   ├── precomputed_dataset_loader.py
│   ├── guitar_tab_collate.py
│   ├── split_dataset.py
│   ├── tab_transformer_model.py
│   ├── tab_transformer_training.py
│   ├── generate_and_precompute.py   # feature extraction + token generation
│   ├── train_transformer.py         
│   ├── evaluate.py                  # load checkpoint & render TAB
│   ├── guitar_tab_vocab_with_extended_techniques.json
│   └── note_tab.json               # custom JAMS namespace schema
├── requirements.txt
└── README.md
```
---

## 🚀 Quick start

Make sure that the paths in the code are correct.

### 1  Set up environment

```bash
git clone https://github.com/georgus92/GuitarTabTransformer.git
cd GuitarTabTransformer

python -m venv .venv        # or conda create ‑n gtt python=3.10
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2  Generate CQT + tokens

```bash
python scripts/generate_and_precompute.py
```

This writes `*.pt` tensors (CQT) and token files into `precomputed/`.

### 3  Split dataset

```bash
python scripts/split_dataset.py
```

### 4  Train

```bash
python train_transformer_amp.py --batch_size 4 --epochs 40 --lr 1e-4
```

### 5  Evaluate and render TAB

```bash
python scripts/evaluate.py
```

The script prints a 6‑line ASCII TAB and saves a `.txt` file next to the audio.

---

## 🛠 Requirements

* Python ≥ 3.10  
* PyTorch ≥ 2.2 + torchaudio  
* librosa, JAMS, NumPy, Matplotlib, Seaborn  
* (Optional) CUDA‑enabled GPU for training

---

## 🔬 Reproducibility checklist

1. Download the full SynthTab dataset.  
2. Run `generate_and_precompute.py` to build the precomputed chunk cache.  
3. Train with the hyper‑parameters batch_size=4 epochs=40 lr=1e-4.  
4. Evaluate on the held‑out test split.  

Every step is scripted; no notebook magic required.

