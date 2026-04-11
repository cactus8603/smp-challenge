# SMP Challenge Project (Multimodal Popularity Prediction)

---

# Overview

This project aims to build a multimodal model for predicting social media popularity (SMP Challenge).

Current focus:
- Strong **text + metadata baseline**
- Modular design for future multimodal extension (image branch)

---

# Project Structure

```text
smp-challenge/
├── configs/
│   ├── base.yaml
│   └── text_meta_cross_v1.yaml
├── data/
│   ├── raw/
│   └── processed/
│       ├── train.parquet
│       └── val.parquet
├── outputs/
│   ├── checkpoints/
│   ├── tensorboard/
│   └── <exp_name>/
├── src/
│   ├── datasets/
│   │   ├── smp_datasets.py
│   │   └── metadata_preprocessor.py
│   ├── models/
│   │   ├── text_encoder.py
│   │   ├── meta_encoder.py
│   │   ├── fusion.py
│   │   ├── fusion_model.py
│   │   └── head.py
│   ├── engine/
│   │   ├── trainer.py
│   │   └── evaluator.py
│   └── utils/
├── scripts/
│   └── train.py
├── build_dataset.py
└── README.md
```

---

# Data Pipeline

## Input Data

Processed dataset (from `build_dataset.py`):

- train.parquet
- val.parquet

Each row represents a post:
- Key: (Uid, Pid)

---

## Metadata Processing

Handled by:
`MetadataPreprocessor`

Feature types:

### Numerical
- time features (hour, weekday, etc.)
- user statistics
- geo features

### Categorical
- category
- subcategory
- concept
- user_id related fields

### Binary
- ispublic
- ispro
- has_geo

Output:

```
meta_num  ∈ R^N
meta_cat  ∈ Z^C
meta_bin  ∈ R^B
```

---

## Dataset (SMPDataset)

Outputs:

```
{
  input_ids,
  attention_mask,
  meta_num,
  meta_cat,
  meta_bin,
  labels
}
```

Text construction:
- full_text (priority)
- title / tags fallback
- topic: category + concept

Tokenizer:
- CLIP tokenizer
- max_length = 77

---

# Model Architecture

## Overview

Current model: Text + Metadata

```
Text Input
   ↓
CLIP Text Encoder
   ↓
text_repr (H)

Metadata
   ↓
MetaEncoder
   ↓
meta_repr (H)

Interaction:
cross = text_repr ⊙ meta_repr
cross = projection

Fusion:
concat(text, meta, cross)
→ MLP

Head:
→ regression output
```

---

## Components

### Text Encoder
- Model: CLIPTextModelWithProjection
- Output: text_repr ∈ R^H
- Default: frozen

---

### Metadata Encoder

Structure:
- numerical branch (MLP)
- categorical embedding
- binary branch

Output:
- meta_repr ∈ R^H

---

### Fusion (Cross Feature)

Explicit interaction:

```
cross = text ⊙ meta
```

Then:

```
fused = concat(text, meta, cross)
→ fusion MLP
```

Reason:
- captures interaction
- lightweight
- better than simple concat

---

### Head

Regression head:
- MLP
- Output: scalar popularity score

---

# Training Pipeline

Script:
```
scripts/train.py
```

Steps:
1. Load config (with base inheritance)
2. Load parquet data
3. Fit metadata preprocessor
4. Build datasets & dataloaders
5. Build model
6. Train + validate
7. Save checkpoints and logs

---

# Config System (YAML)

Supports base inheritance:

```yaml
base: base.yaml

train:
  lr: 2e-5
  epochs: 8
```

Sections:

- data
- model
- text
- meta
- fusion
- train
- output

---

# Current Status

Implemented:
- text encoder (CLIP)
- metadata encoder
- cross-feature fusion
- training pipeline
- config system

Not implemented:
- image encoder
- ranking loss
- advanced fusion (attention)

---

# TODO

## Immediate
- run full training
- verify pipeline end-to-end
- baseline comparison:
  - metadata-only
  - text-only
  - text + metadata

---

## Feature Engineering
- tag target encoding
- category / concept encoding
- user-level statistics

---

## Model Improvements
- compare fusion methods
- add text cluster features
- add lightweight text features

---

## Image Branch
- implement image encoder
- extend fusion to multimodal

---

## Advanced
- ranking-aware loss
- cross-attention
- user modeling

---

# Notes

- metadata is highly important in this task
- text alone is not sufficient
- interaction between text and metadata is critical
