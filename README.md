# SMP Challenge Project (Multimodal Popularity Prediction)

This project aims to build a multimodal model for predicting social media popularity (SMP Challenge).

---

# Current Progress (2026-04)

We have completed the full training pipeline backbone (without image branch yet).

## Completed Modules

### 0. Data
```text
smp-challenge/
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── checkpoints/
│   ├── tensorboard/
│   └── experiments/
│       └── exp_001/
│           ├── config.yaml
│           ├── log.txt
│           └── result.json
├── src/
│   ├── datasets/
│   ├── models/
│   ├── utils/
│   └── engine/
│       ├── trainer.py
│       └── evaluator.py
├── scripts/
│   ├── train.py
│   ├── infer.py
│   └── eval.py
├── build_dataset.py
├── README.md
├── requirements.txt
└── .gitignore
```

### 1. Dataset Processing
- Built build_dataset.py
- Merged raw SMP data into unified tables
- Key design:
  - Post-level key: Uid + Pid
  - User-level join: Uid
- Output formats:
  - train.parquet (main training data)
  - test.parquet
  - jsonl / csv (for debugging)

---

### 2. Data Pipeline (SMPDataset)
- Located at: src/datasets/smp_dataset.py
- Responsibilities:
  - Load parquet data
  - Build text input (full_text + category + tags)
  - Extract metadata features
  - Tokenize text (HuggingFace tokenizer)
  - Return:
    - input_ids
    - attention_mask
    - meta_features
    - labels

- Image branch:
  - Placeholder only (not implemented yet)

---

### 3. Metadata Features

Currently used:

Time:
- hour, weekday, is_weekend, year, month, day

User:
- photo_count (log1p)
- ispro, canbuypro
- timezone_offset, timezone_id

Geo:
- latitude, longitude, geoaccuracy, has_geo

System:
- ispublic

Note:
Metadata is very important for this task.

---

### 4. Config System (YAML-based)
- Located at: configs/
- Supports:
  - Multiple experiments
  - Base config inheritance

Example:

python src/main.py --config configs/text_meta_v1.yaml

---

### 5. Training Pipeline (main.py)
- Full training loop implemented:
  - Train / Validation
  - Checkpoint saving
  - Best model selection (by Spearman)

- Supports:
  - Text + Metadata
  - Optional image branch (future)

---

### 6. Metrics (metrics.py)

Implemented:
- Spearman Correlation (main metric)
- MAE
- MSE, RMSE

Important:
- Model selection is based on Spearman, not loss.

---

### 7. TensorBoard Logging

Logs:
- train loss
- validation loss
- MAE
- Spearman
- learning rate

Run:

tensorboard --logdir outputs/tensorboard

---

### 8. Experiment Management

Each run will automatically save:

outputs/
  checkpoints/<exp_name>/
    best.pt
    latest.pt
  tensorboard/<exp_name>/
  <exp_name>/
    config.json
    history.json
    summary.json

---

# Current Model Scope

Implemented:
- Text encoder (transformer)
- Metadata encoder (MLP)

Not yet implemented:
- Image encoder
- Fusion module
- Ranking-based loss

---

# Observations from Data

- Same full_text can appear many times for the same user
- Text is not always unique per post
- Metadata (especially user + time) is critical

---

# Design Philosophy

- Modular design
- Easy experiment control (YAML)
- Reproducibility (config saving)
- Fast iteration (no image dependency initially)

---

# Next Steps

Short-term:
- Implement fusion_model.py
- Run baseline:
  - metadata-only
  - text-only
  - text + metadata

Mid-term:
- Add image encoder
- Improve feature engineering
- Add normalization for metadata

Advanced:
- Ranking-aware loss
- Better fusion (cross-attention)
- User-level modeling

---

# How to Run

python src/main.py --config configs/text_meta_v1.yaml

---

# Notes

- Start simple (text + metadata)
- Gradually evolve into multimodal model
- Focus on strong baselines first
