# Experiment Results

Metric: **Spearman correlation** on fold-0 validation set (GroupKFold by Uid, 5 folds).

---

## Single-model experiments

| Exp | Description | Val Spearman | Best Epoch | Notes |
|-----|-------------|:------------:|:----------:|-------|
| test_v3 | Joint (text + image + meta), hybrid loss | 0.6220 | — | baseline joint model |
| test_v5 | test_v3 + sentiment fix in processed_v4 | 0.6220 | — | sentiment bug fixed, no gain |
| test_v8 | Metadata MLP only (no text/image), processed_v4 | 0.5104 | 4 | pure metadata branch; LightGBM proxy 0.5161 |
| test_v9 | Image + text + 3 CLIP sim features, no traditional meta | 0.5975 | 4 | processed_v5 data; concat fusion; peaked early |
| test_v10 | Joint (text + image + meta + 3 CLIP sim), processed_v5, pairwise_gated | 進行中 | — | epoch 4 達 0.6082，共 13 epochs |

### test_v9 preprocess features
- `clip_text_image_sim`: cos_sim(CLIP(title+tags+topic), CLIP(image))
- `clip_text_caption_sim`: cos_sim(CLIP(user_text), CLIP(blip2_caption))
- `clip_caption_image_sim`: cos_sim(CLIP(blip2_caption), CLIP(image))

---

## Ensemble experiments

### ensemble_v8_v9 (2026-05-16)

Script: `scripts/ensemble_v8_v9.py`
Formula: `alpha * preds_v9 + (1 - alpha) * preds_v8`

| alpha (v9 weight) | Val Spearman |
|:-----------------:|:------------:|
| 0.00 (v8 only) | 0.5104 |
| 0.10 | 0.5360 |
| 0.20 | 0.5599 |
| 0.30 | 0.5803 |
| 0.40 | 0.5965 |
| 0.50 | 0.6079 |
| 0.60 | 0.6141 |
| **0.70** | **0.6156** ← best |
| 0.80 | 0.6128 |
| 0.90 | 0.6065 |
| 1.00 (v9 only) | 0.5975 |

**Best ensemble: alpha=0.70, spearman=0.6156** (+0.0181 vs v9 alone)

Best ensemble: alpha=0.70, spearman=0.6156 (+0.0181 vs v9 alone)

Still below joint training (test_v3/v5: 0.6220). Confirms cross-modal gradients matter.

---

## Stacking experiments

### test_v11 — Frozen encoder stacking (2026-05-17)

Script: `scripts/train_v11.py`

架構：
- v8 encoder（frozen）→ meta_repr（256-dim）
- v9 encoder（frozen）→ text_repr（256-dim）+ image_repr（256-dim）
- concat → 768-dim → MLP fusion head（768→512→256→1，60 epochs）

| | Val Spearman |
|--|:------------:|
| v8 standalone | 0.5104 |
| v9 standalone | 0.5975 |
| weighted ensemble (alpha=0.70) | 0.6156 |
| test_v5 joint training (舊 baseline) | 0.6220 |
| **test_v11 frozen stacking** | **0.6315** |

**超越 joint training baseline (+0.0095)**。兩個 encoder 各自充分訓練後，MLP fusion head 能學到更好的非線性組合。

---

## Data versions

| Version | Description |
|---------|-------------|
| processed_v2 | baseline with user_desc embeddings |
| processed_v4 | + BLIP-2 captions, sentiment fix |
| processed_v5 | + 3 CLIP similarity features (from processed_v4) |

---

## Next

- **test_v10**（進行中）：Joint model (text + image + meta + 3 CLIP sim)，等結果
- **test_v11 延伸**：在 v11 stacking 基礎上，加入 test_v10 encoder（訓練完後）→ 三個 encoder stacking
- **test_v12 候選**：同學建議的 hierarchical fusion + meta 殘差，修改 pairwise_gated
