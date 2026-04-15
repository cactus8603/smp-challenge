# Factorized Multimodal Popularity Prediction
**Working Design Document — SMP Challenge 2025**

---

## 0. 背景與動機

### 問題定義

Social Media Popularity Prediction（SMP）的目標是預測一篇貼文的 log(views)。

現有方法（包括論文 HyperFusion）的主要做法：
```
multimodal features → concat → MLP → score
```

這種做法讓模型隱式地學習所有因素的交互，但有幾個問題：
- 不 interpretable：不知道哪個因素影響最大
- 泛化差：不同 domain 的 factor 重要性不同，但模型無法調整
- 對 ranking 不友善：MLP 優化 regression，不直接優化 Spearman

### 核心想法

> 熱門度不是單一訊號，而是多個因素交互的結果。

明確建模這些因素，而不是讓單一 MLP 隱式學習。

---

## 1. Factorization 設計

### 四個 Latent Factor

| Factor | 代號 | 定義 | 主要輸入 |
|--------|------|------|----------|
| Appeal | A | 內容本身的吸引力 | image + text |
| Exposure | E | 時空/類別條件 | temporal + geo + category |
| User Prior | U | 用戶歷史表現 | user metadata |
| Context Matching | C | 內容與情境的匹配度 | content × metadata |

### 預測公式

基礎版：
```
y_hat = f(A, E, U, C)
```

帶 interaction 的完整版：
```
y_hat = b
      + w_A·A + w_E·E + w_U·U + w_C·C
      + w_AE·(A ⊙ E)
      + w_AU·(A ⊙ U)
      + w_CE·(C ⊙ E)
```

---

## 2. 架構設計

### 2.1 Encoder 層

#### Content Encoder
```
image  → CLIP Vision Encoder  → z_img  ∈ R^512
text   → CLIP Text Encoder    → z_text ∈ R^512
z_content = proj([z_img, z_text]) ∈ R^d
```

**注意**：CLIP 輸出是 L2-normalized，進 fusion 之前要先 normalize 再做 element-wise 乘法，否則數值意義不正確。

#### Metadata Encoder
```
inputs: category, geo, temporal, stats
z_meta = MetaEncoder(num, cat, bin) ∈ R^d
```

MetaEncoder 已有實作，使用 gated fusion 合併 numeric / categorical / binary 三個 branch。

#### User Encoder（v1 簡化版）
```
z_user = MLP_U([user_mean_label, follower_count,
                total_views, photo_count, ...]) ∈ R^d
```

**v2 真正的 User Encoder**（之後再做）：
```
z_user = AttentionPool({z_content_i, z_meta_i} for i in user_history)
```
用 DeepSets 或 attention pooling 編碼用戶歷史貼文，捕捉用戶的內容風格偏好。

---

### 2.2 Factor Head 層

#### Appeal Head (A)
```python
A = MLP_A(z_content)  # 純內容吸引力
```

#### Exposure Head (E)
```python
E = MLP_E(z_meta)  # 時空/類別條件
```

#### User Prior Head (U)
```python
U = MLP_U(z_user)  # 用戶影響力
```

#### Context Matching Head (C) ⭐ 主要 Novelty

明確建模 content 和 context 的對齊程度。

**特徵構造**：
```python
concat_feat   = torch.cat([z_content, z_meta], dim=-1)   # [B, 2d]
product_feat  = z_content * z_meta                        # [B, d]  element-wise
diff_feat     = torch.abs(z_content - z_meta)             # [B, d]  絕對差
bilinear_feat = z_content * (W_bilinear @ z_meta.T).T     # [B, d]  bilinear（向量形式）

C_input = torch.cat([concat_feat, product_feat, diff_feat, bilinear_feat], dim=-1)
C = MLP_C(C_input)  # [B, factor_dim]
```

**詮釋**：
- `product`：捕捉共同激活的維度（兩者都高才高）
- `diff`：捕捉不匹配的維度
- `bilinear`：學習任意的 cross-modal 交互

**改進點（待探索）**：
- C 應該是 scalar 還是 vector？scalar 更 interpretable，vector 更有表達力
- bilinear 矩陣 W 的 rank 要多大？可以用 low-rank approximation 省參數

---

### 2.3 Interaction Layer

用 **structured interaction** 取代 full attention，計算成本低且可解釋。

```python
# input-dependent gating（比固定權重強）
gate_AE = torch.sigmoid(MLP_gate_AE(torch.cat([z_A, z_E], dim=-1)))
gate_AU = torch.sigmoid(MLP_gate_AU(torch.cat([z_A, z_U], dim=-1)))
gate_CE = torch.sigmoid(MLP_gate_CE(torch.cat([z_C, z_E], dim=-1)))

h_AE = gate_AE * (W_A(z_A) * W_E(z_E))  # Appeal × Exposure
h_AU = gate_AU * (W_A(z_A) * W_U(z_U))  # Appeal × User
h_CE = gate_CE * (W_C(z_C) * W_E(z_E))  # Context × Exposure
```

**為什麼選這三個 pair？**
- AE：同樣的內容在不同時間/地點/類別，熱門度不同
- AU：同樣的內容，有大量粉絲的用戶熱門度更高
- CE：內容和情境的匹配度在特定條件下影響更大

---

### 2.4 Prediction Head

```python
h = torch.cat([A, E, U, C, h_AE, h_AU, h_CE], dim=-1)
y_hat = MLP_out(h)
```

**注意**：不要加 Softplus 在最後。原因是初始化時容易讓所有輸出擠在一起（output collapse），直接讓 loss 約束輸出範圍比較穩。

---

## 3. 訓練目標

### 3.1 Regression Loss
```
L_reg = SmoothL1(y_hat, y)
```

### 3.2 Ranking Loss ⭐ 重要

對 pair (i, j)：
```
L_rank = -log σ((y_hat_i - y_hat_j) · sign(y_i - y_j) - margin)
```

**Pair sampling 策略**（重要性由高到低）：

1. **Same-user pairs**（最重要）：排除 user influence，讓模型專注 content/context 差異
2. **Same-category pairs**：控制 category 變數
3. **Random global pairs**：噪音較多，權重要低

```python
def sample_pairs(labels, uids, max_pairs=1024):
    # 優先 same-user pairs
    # 再補 same-category pairs
    # 最後加少量 random pairs
    ...
```

**注意**：`max_pairs` 要設上限，避免 O(B²) 記憶體爆炸。batch_size=256 時，256² = 65536 pairs，佔用大量 GPU memory。

### 3.3 Factor Regularization（可選）

鼓勵 factor 之間解耦。

**不推薦用 correlation loss**（batch 估計太噪）：
```python
# ❌ 不穩定
L_factor = Σ corr(z_i, z_j)^2
```

**推薦用 VICReg covariance loss**：
```python
def covariance_loss(z):
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (z.size(0) - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag ** 2).sum() / z.size(1)

L_factor = covariance_loss(A) + covariance_loss(E) + covariance_loss(U) + covariance_loss(C)
```

### 最終 Loss
```
L = λ1·L_reg + λ2·L_rank + λ3·L_factor

建議初始值：λ1=1.0, λ2=0.2, λ3=0.01
```

---

## 4. 待探索問題

| 問題 | 目前傾向 | 原因 |
|------|----------|------|
| C 是 scalar 還是 vector？ | vector | 表達力更強，scalar 資訊量太少 |
| bilinear W 的 rank？ | 低 rank（16~32） | 省參數，避免 overfitting |
| Ranking pairs 的比例？ | same-user 60%, same-cat 30%, random 10% | 待 ablation |
| Factor normalization？ | L2 normalize 各 factor | 讓 interaction 在同一 scale |
| λ 的調整策略？ | 先固定 λ2=0.2，跑 ablation | 避免 rank loss 壓制 reg loss |

---

## 5. 實作計畫

### Phase 1（目前，先驗證核心想法）
- [ ] 實作 `ContextMatchingHead`
- [ ] 把現有 MetaEncoder / TextEncoder 接成 Factor Head
- [ ] 改 ranking loss 為 same-user pair sampling
- [ ] 整合進 `fusion_model.py`
- [ ] 跑 ablation：有無 C head 的差距

### Phase 2（Phase 1 驗證後）
- [ ] 加 interaction gating（input-dependent）
- [ ] 加 factor regularization（VICReg 版）
- [ ] 調整 λ 參數

### Phase 3（最後衝分）
- [ ] True User Encoder（attention pooling on user history）
- [ ] Prototype refinement（prototype bank）
- [ ] 5-fold + ensemble

---

## 6. 預期優勢

| 優勢 | 說明 |
|------|------|
| 更好的 Spearman | Ranking loss 直接優化排序 |
| Interpretability | 可以分析每個 factor 的貢獻 |
| 泛化能力 | 結構化的因素分解比純 MLP 更 robust |
| Research novelty | Context Matching 是現有論文沒有明確建模的部分 |

---

## 7. 注意事項

### Output Collapse 問題
模型訓練初期（第 1~2 epoch）pred_std 很小是正常的，模型在預測均值。
判斷標準：
```
Epoch 5 之後 pred_std 仍 < 0.5 → 真正的 collapse，需要調整
Epoch 5 之後 pred_std > 1.0   → 正常學習
```

### 先把 Baseline 跑穩
在實作這個複雜架構之前，先確認現有的 text + meta baseline 已經在正常學習（pred_std > 1.5）。否則無法判斷新架構的提升是來自 factorization 還是修好了其他問題。

### CLIP 的特性
CLIP 輸出是 L2-normalized，做 element-wise 乘法前需要確保 meta encoder 的輸出也在同一 scale，否則乘法的數值意義不正確。

---

## 8. 參考

- HyperFusion（你在參考的論文）：CatBoost + TabNet + MLP ensemble，3rd place SMP 2025
- AMCFG（另一篇 SMP 2025 論文）：clustering-based anchoring，用 LLM 生成 semantic anchor
- FM / DeepFM：factorization machine 的理論基礎
- VICReg：variance-invariance-covariance regularization，factor disentanglement 的參考

---

*最後更新：2026-04-15*