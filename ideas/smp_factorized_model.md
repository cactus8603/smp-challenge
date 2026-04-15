# Factorized Multimodal Popularity Prediction (Working Draft)

## 0. Goal
Build a **research-level model** for Social Media Popularity Prediction (SMP) that goes beyond standard multimodal fusion.

Key idea:
> Popularity is not a single signal, but the result of multiple interacting factors.

We explicitly model these factors instead of letting a single MLP learn everything implicitly.

---

## 1. Core Idea: Factorization

We decompose popularity into four latent factors:

- **A (Appeal)**: intrinsic content quality (image + text)
- **E (Exposure)**: temporal / spatial / category conditions
- **U (User prior)**: user's historical performance
- **C (Context Matching)**: how well content matches its context

Final prediction:

```
y_hat = f(A, E, U, C)
```

Expanded form (example):

```
y_hat = b \
      + w_A * A + w_E * E + w_U * U + w_C * C \
      + w_AE * (A ⊙ E) \
      + w_AU * (A ⊙ U) \
      + w_CE * (C ⊙ E)
```

---

## 2. Overall Architecture

### Stage 1: Encoders

#### 1. Content Encoder
- Image: CLIP Vision Encoder
- Text: CLIP Text Encoder
- Tag embeddings (optional)

Output:
```
z_content ∈ R^d
```

---

#### 2. Metadata Encoder
Includes:
- category / subcategory / concept
- geo (lat, lon)
- temporal (hour, weekday)
- additional stats

Output:
```
z_meta ∈ R^d
```

---

#### 3. User Encoder (optional but recommended)

Input:
- user's historical posts (content + metadata)

Method:
- Set encoder (DeepSets / attention pooling)

Output:
```
z_user ∈ R^d
```

---

## 3. Factor Heads

### 3.1 Appeal Head (A)

```
A = MLP_A(z_content)
```

Represents intrinsic attractiveness.

---

### 3.2 Exposure Head (E)

```
E = MLP_E(z_meta)
```

Represents platform conditions:
- timing
- location
- category popularity

---

### 3.3 User Prior Head (U)

```
U = MLP_U(z_user)
```

Captures:
- average popularity
- stability
- user influence

---

### 3.4 Context Matching Head (C) ⭐ KEY NOVELTY

We explicitly model **content-context alignment**.

Inputs:
```
z_content, z_meta
```

Feature construction:
```
concat = [z_content, z_meta]
product = z_content ⊙ z_meta
diff = |z_content - z_meta|
bilinear = z_content^T W z_meta
```

Then:
```
C = MLP_C([concat, product, diff, bilinear])
```

Interpretation:
- high when content matches time/location/category
- low when mismatched

---

## 4. Interaction Layer

Instead of full attention, we use **structured interactions**:

```
h_AE = (W_A z_A) ⊙ (W_E z_E)
h_AU = (W_A z_A) ⊙ (W_U z_U)
h_CE = (W_C z_C) ⊙ (W_E z_E)
```

Optional gating:

```
h_AE = gate_AE * h_AE
```

---

## 5. Prediction Head

Final input:
```
h = [A, E, U, C, h_AE, h_AU, h_CE]
```

Prediction:
```
y_hat = MLP_out(h)
```

---

## 6. Training Objective

### 6.1 Regression Loss

```
L_reg = SmoothL1(y_hat, y)
```

---

### 6.2 Ranking Loss ⭐ IMPORTANT

For pair (i, j):

```
L_rank = -log σ((y_hat_i - y_hat_j) * sign(y_i - y_j))
```

Pair sampling strategy:
- same user, different posts
- same category, different time
- random global pairs

---

### 6.3 Factor Regularization (optional)

Encourage disentanglement:

```
L_factor = Σ corr(z_i, z_j)^2
```

(for i ≠ j)

---

### Final Loss

```
L = λ1 * L_reg + λ2 * L_rank + λ3 * L_factor
```

---

## 7. (Optional) Prototype Refinement (v2+)

Maintain prototype bank:

```
P = {p1, p2, ..., pk}
```

Compute similarity:

```
w_k = softmax(q · p_k)
r = Σ w_k p_k
```

Refine representation:

```
z' = z + MLP(r)
```

Can apply on:
- content
- factor space

---

## 8. Key Contributions (for Paper)

1. Factorized popularity modeling
2. Content-context matching mechanism
3. Factor-aware ranking objective
4. Structured interaction instead of naive fusion

---

## 9. Implementation Plan

### v1 (baseline research model)
- encoders (reuse existing)
- add Context Matching Head
- factorized prediction head
- ranking loss

### v2
- user style encoder
- interaction gating
- factor regularization

### v3
- prototype refinement

---

## 10. Notes / To Explore

- Should C be scalar or vector?
- Better bilinear design?
- Ranking sampling strategies
- Whether to normalize factors
- Loss balancing (λ tuning)

---

## 11. Expected Advantages

- Better alignment with Spearman
- Interpretability (factor analysis)
- Stronger generalization
- Clear research novelty

---

## 12. TODO

- [ ] implement ContextMatchingHead
- [ ] implement FactorHead
- [ ] implement RankingLoss
- [ ] integrate into fusion_model.py
- [ ] run ablation study

---

(End of draft)

