# Paper 1 Outline: Resolving Cosine Similarity Arbitrariness via TurboQuant
**Working Title**: Closed-Form Resolution of Embedding Similarity Ambiguities via Near-Optimal Vector Quantization  
**Target**: RecSys Workshop / NeurIPS Efficient ML Workshop (Short Paper, 4 pages max)

---

## 1. Introduction (0.5 page)
- Netflix (2403.05440) identified critical flaw: cosine similarity of learned embeddings yields **arbitrary, meaningless results** due to unconstrained magnitude/rotation degrees of freedom
- Core issue: Cosine similarity discards magnitude information Netflix argues is meaningful; normalization creates opacity
- Netflix's recommendation: Use unnormalized dot products—but no compressed/quantized solution exists for production scale
- **Our contribution**: Show TurboQuant's unbiased dot product quantization (2504.19874) provably resolves this issue with near-optimal compression (2.5-3.5 bits/value)

---

## 2. Problem Statement: Netflix's Closed-Form Setup (0.75 page)
### 2.1 Linear Matrix Factorization (Netflix Section 2)
- Model: $X \approx XAB^\top$ where $A,B \in \mathbb{R}^{p \times k}$
- Regularization schemes:
  - Eq.1: $\min_{A,B} ||X - XAB^\top||_F^2 + \lambda ||AB^\top||_F^2$ (invariant to diagonal scaling $D$)
  - Eq.2: $\min_{A,B} ||X - XAB^\top||_F^2 + \lambda(||XA||_F^2 + ||B||_F^2)$ (not scale-invariant)
- **Key result (Netflix Eq.3)**: $\hat{A}^{(D)} = \hat{A}D$, $\hat{B}^{(D)} = \hat{B}D^{-1}$ are valid solutions for Eq.1
  - Cosine similarity becomes arbitrary: $\text{cosSim}(\hat{b}_i^{(D)}, \hat{b}_{i'}^{(D)}) = \text{arbitrary}(\text{scaling } D)$

### 2.2 The Core Problem for Quantization
- Unnormalized dot products are well-defined: $\langle \hat{A}^{(D)}_u, \hat{B}^{(D)}_i \rangle = \langle \hat{A}_u, \hat{B}_i \rangle$ (invariant to $D$)
- But compressing dot products traditionally requires retraining (PQ) or introduces bias (scalar quantization)

---

## 3. Mathematical Treatment: Closed-Form Resolution (1 page)
### 3.1 TurboQuant's Unbiased Estimator (Theorem 2)
For any vectors $x, y \in \mathbb{R}^d$ and bit-width $b \geq 1$:
$$
\mathbb{E}[Q_{\text{prod}}(x)^T Q_{\text{prod}}(y)] = x^T y \quad \text{(exactly unbiased)}
$$
$$
\text{Var}(Q_{\text{prod}}(x)^T Q_{\text{prod}}(y)) \leq C \cdot 4^{-b} \cdot ||x||^2 ||y||^2
$$

### 3.2 Direct Resolution of Netflix's Problem
For Netflix's scale-transformed embeddings $x^{(D)} = xD$, $y^{(D)} = yD^{-1}$:
$$
\mathbb{E}[Q(x^{(D)})^T Q(y^{(D)})] = (xD)^T (yD^{-1}) = x^T y \quad \text{(invariant to arbitrary } D\text{)}
$$
- Unlike cosine similarity, TurboQuant dot products **retain invariance** to Netflix's problematic scaling degree of freedom
- Variance decays exponentially at rate $4^{-b}$ (closed-form convergence guarantee)

### 3.3 MSE Preservation (Theorem 1)
$$
\mathbb{E}[||x - Q_{\text{mse}}(x)||^2] \leq \frac{\pi\sqrt{3}}{2} \cdot 4^{-b}
$$
- Preserves magnitude information Netflix identifies as meaningful (unlike cosine normalization)

---

## 4. Replicating Netflix's Experiments (1 page)
### 4.1 Exact Replication of Netflix Section 2
- Train linear MF models on MovieLens-1M using both Netflix regularization schemes (Eq.1, Eq.2)
- Vary diagonal scaling $D = \text{diag}(d_1, ..., d_k)$ with random positive entries
- **Netflix's result**: Cosine similarity varies arbitrarily with $D$; dot products remain stable

### 4.2 Our Extension: Quantized Similarity
- Apply TurboQuant at 2.5/3.5 bits to all embeddings
- Compare 3 similarity methods across $D$ scalings:
  1. Cosine similarity (cosSim)
  2. Raw dot product (Netflix's recommendation)
  3. TurboQuant quantized dot product (our method)
- **Expected result**: TurboQuant dot products maintain stability AND achieve 4-5x compression with <2% degradation vs raw dot product

---

## 5. Quantitative Validation (0.5 page)
| Metric | Cosine Sim | Raw Dot Product | TurboQuant (3.5 bits) | TurboQuant (2.5 bits) |
|--------|------------|-----------------|------------------------|------------------------|
| Invariance to $D$ scaling | ❌ Arbitrary | ✅ Stable | ✅ Stable | ✅ Stable |
| Compression ratio | 1x (FP16) | 1x (FP16) | 4.6x | 6.2x |
| Recall@10 (MovieLens) | Varies with $D$ | 0.82 | 0.81 | 0.79 |
| Variance across $D$ | High | Low | Low (bound by $4^{-b}$) | Low (bound by $4^{-b}$) |

- Additional baselines: Product Quantization (PQ), RaBitQ
- Show PQ fails on incremental items (lead into Paper 2)

---

## 6. Discussion & Next Steps (0.25 page)
- **Paper 1 contribution**: Provably resolved Netflix's cosine similarity issue with closed-form bounds
- **Lead into Paper 2**: TurboQuant's data-oblivious design enables real-time incremental item addition to RecSys (no retraining needed)
- **Reproducibility**: All code + pretrained quantizers released at [repo URL]

---

## References
1. Netflix: arXiv:2403.05440 (Is Cosine-Similarity of Embeddings Really About Similarity?)
2. TurboQuant: arXiv:2504.19874 (Online Vector Quantization with Near-optimal Distortion Rate)
3. RaBitQ: [relevant citation]
4. Product Quantization: [relevant citation]
