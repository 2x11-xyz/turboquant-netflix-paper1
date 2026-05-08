# Paper 1: Preserving Scale-Invariant Scores Under Compression
**Working Title**: Preserving Scale-Invariant User-Item Scores via Near-Optimal Quantization  
**Target**: RecSys Workshop / NeurIPS Efficient ML Workshop (Short Paper, 4 pages max)

---

## 1. Introduction (0.5 page)
- Netflix (2403.05440) proved cosine similarity of learned embeddings is **arbitrary**: equivalent models under diagonal rescaling D produce different cosine values, so cosine-based comparisons are meaningless
- Their fix: use raw dot products, which are provably D-invariant. **This is theoretically correct and settled.**
- **The production gap**: Raw dot products solve the invariance problem, but serving them at scale requires compression. Systems with billions of items cannot store full-precision embeddings — they need compressed representations for memory and ANN retrieval (FAISS, ScaNN, etc.)
- **The compression dilemma**: Common deployed schemes do not provide invariance-preserving guarantees:
  - Deterministic scalar quantization introduces **bias** (systematic shift in estimated dot products)
  - Product Quantization (PQ) trains **data-dependent codebooks** that may require retraining as model or data distributions shift
  - Neither guarantees that the compressed dot product equals the true dot product in expectation
- **Our contribution**: We observe that TurboQuant (2504.19874), a near-optimal multi-bit unbiased inner product quantizer, preserves Netflix's D-invariant scores in expectation at 2–3 bits/value. We make this connection explicit, verify it experimentally on Netflix's own synthetic setup, and derive a prescriptive link between Eq.2 weight decay and quantization variance control. Our theoretical observation is a corollary of Netflix Eq.3 and TurboQuant Theorem 2; the contribution is making this connection explicit and testing its implications for recommender compression.

*Note: TurboQuant is not the first unbiased compressed estimator for dot products — QJL (Zandieh et al., 2024) provides unbiased 1-bit estimation. TurboQuant's novelty is achieving near-optimal distortion rates at arbitrary bit-widths.*

---

## 2. Problem Statement: Netflix's Closed-Form Setup (0.75 page)
### 2.1 Linear Matrix Factorization (Netflix Section 2)
- Model: $X \approx XAB^\top$ where $A,B \in \mathbb{R}^{p \times k}$
- User embeddings $U = XA \in \mathbb{R}^{n \times k}$, item embeddings $V = B \in \mathbb{R}^{p \times k}$
- Regularization schemes:
  - **Eq.1**: $\min_{A,B} \|X - XAB^\top\|_F^2 + \lambda \|AB^\top\|_F^2$ — invariant to diagonal scaling D
  - **Eq.2**: $\min_{A,B} \|X - XAB^\top\|_F^2 + \lambda(\|XA\|_F^2 + \|B\|_F^2)$ — breaks D-invariance via weight decay

### 2.2 The D-Scaling Ambiguity (Netflix Eq.3)
- For any invertible diagonal $D$: $\hat{A}^{(D)} = \hat{A}D$, $\hat{B}^{(D)} = \hat{B}D^{-1}$ are equivalent Eq.1 minimizers
- **Cosine similarity is arbitrary**: $\cos(u^{(D)}, v^{(D)})$ changes with D
- **Dot product is invariant**: $\langle u^{(D)}, v^{(D)} \rangle = \langle uD, vD^{-1} \rangle = \langle u, v \rangle$ for all D
- Netflix's recommendation is correct — but silent on how to **compress** these dot products for production serving

### 2.3 Why Compression Matters
- Full-precision storage is expensive at billion-item scale, and common compressed serving pipelines do not provide invariance-preserving guarantees
- Standard quantization Q can introduce bias: $\mathbb{E}[\langle u, Q(v) \rangle] \neq \langle u, v \rangle$
- PQ trains data-dependent codebooks that may require recalibration as distributions shift
- **Result**: The correct similarity measure (dot product) needs a compression method with formal unbiasedness guarantees

---

## 3. TurboQuant Preserves D-Invariance Under Compression (1 page)
### 3.1 One-Sided Unbiased Estimator (TurboQuant Theorem 2)
TurboQuant_IP quantizes **one argument** (items) while leaving the other (users) in full precision. For unit-norm $x \in S^{d-1}$ (with norms stored separately and restored via the standard norm-storage extension described in the TurboQuant paper):
$$\mathbb{E}[\langle y, \text{DeQuant}(Q(x)) \rangle] = \langle y, x \rangle \quad \text{(exactly unbiased)}$$

Applied to Netflix's D-scaled embeddings (quantize items only):
$$\mathbb{E}[\langle u^{(D)}, \text{DeQuant}(Q(v^{(D)})) \rangle] = \langle u^{(D)}, v^{(D)} \rangle = \langle u, v \rangle$$

The compressed estimator preserves D-invariance **in expectation over quantization randomness** at 2–3 bits/value.

### 3.2 Variance Depends on κ(D) — The Prescriptive Link
The variance of the quantized inner product depends on **both** the query norm and the quantized item norm:
$$\text{Var}[\langle u^{(D)}, \hat{v}^{(D)} \rangle] \leq \frac{\sqrt{3}\pi^2}{d} \cdot 4^{-b} \cdot \|u^{(D)}\|^2 \cdot \|v^{(D)}\|^2 = \frac{\sqrt{3}\pi^2}{d} \cdot 4^{-b} \cdot \|uD\|^2 \cdot \|vD^{-1}\|^2$$

Under D-scaling with condition number $\kappa(D) = d_{\max}/d_{\min}$:
$$\|uD\| \leq d_{\max}\|u\|, \quad \|vD^{-1}\| \leq d_{\min}^{-1}\|v\|$$

So variance is bounded by $O(\kappa(D)^2 \cdot \|u\|^2\|v\|^2 / d \cdot 4^{-b})$.

This creates a **prescriptive connection** to Netflix's Eq.2 regularization:
- **Eq.2 controls both factors**: the $\|XA\|_F^2$ term bounds user norms $\|u\|^2$, and the $\|B\|_F^2$ term bounds item norms $\|v\|^2$
- This is strictly stronger than Eq.1, which leaves both norms unconstrained under D-rescaling
- Practitioners planning to quantize embeddings should prefer Eq.2-style regularization

### 3.3 MSE Preservation (TurboQuant Theorem 1)
$$\mathbb{E}[\|x - Q_{\text{mse}}(x)\|^2] \leq \frac{\pi\sqrt{3}}{2} \cdot 4^{-b}$$
Unlike cosine normalization, TurboQuant does not explicitly discard norm information; its MSE guarantee provides controlled reconstruction error of the original vector.

---

## 4. Experiments (1 page)

### 4.1 Synthetic Data (Netflix Section 4 Replication)
Following Netflix's exact setup:
- n=20,000 users, p=1,000 items, C=5 clusters with known ground truth
- Power-law item popularity ($\beta_{\text{item}} \in [0.25, 1.5]$) and user activity ($\beta_{\text{user}} = 0.5$)
- Train with Eq.1 ($\lambda=10{,}000$) and Eq.2 ($\lambda=100$), $k=50$
- D-scalings from Netflix Figure 1: $B = V_k$, $B = V_k \cdot \text{dMat}(\sigma_i^2)$, $B = V_k \cdot \text{dMat}((1+\lambda/\sigma_i^2)^{1/2})$, $B = V_k \cdot \text{dMat}(\sigma_i(1-\lambda/\sigma_i^2)_+^{1/2})$

### 4.2 Figure 1: Visual Demonstration (3 rows × 6 columns)
| Row | What it shows | Key finding |
|-----|---------------|-------------|
| **Row 1 (Netflix)** | True clusters + cosSim(B,B) under 4 D-scalings + Eq.2 | Cosine similarity is arbitrary — block structure changes or disappears |
| **Row 2 (Dot product)** | Same layouts, raw dot product matrices | Dot product matrices are identical regardless of D-scaling — D-invariant |
| **Row 3 (Ours)** | Same layouts, TurboQuant 3-bit quantized dot product | Block structure is preserved under compression — TQ maintains the signal |

### 4.3 Quantitative Validation (Table — Eq.1 only)
D-scaling via $D = \text{diag}(e^{tz_1}, \ldots, e^{tz_k})$, $t \in \{0, 0.5, 1, 2, 5\}$

| Metric | What it measures |
|--------|-----------------|
| Mean cosine similarity | Should change with t (arbitrary) |
| Mean dot product | Should be constant across t (D-invariant) |
| TQ mean estimate vs true dot product | **Bias verification**: E[TQ] − ⟨u,v⟩ should be ≈ 0 |
| TQ variance across seeds | Should grow with κ(D) |

Key observations:
1. Dot product is constant across all t — D-invariant ✓
2. Cosine similarity changes — arbitrary under D ✓
3. **TQ mean estimate is unbiased**: E[TQ] ≈ true dot product for all t ✓
4. TQ variance grows with κ(D) as predicted by the bound ✓
5. Eq.2 is not D-scaled (its solution is unique; D-scaling is only meaningful for Eq.1)

---

## 5. Discussion & Next Steps (0.25 page)
- **Contribution**: Netflix identified the right similarity measure (dot product); we show TurboQuant provides a near-optimal compressed estimator that preserves it in expectation. The theoretical observation is a corollary of two existing results; the contribution is making the connection explicit, deriving the variance-κ(D) bound, and validating experimentally.
- **Practical guidance**: Use Eq.2-style weight decay when planning to quantize — it bounds worst-case quantization variance by controlling both user and item embedding norms.
- **Limitation**: Our theory is derived for linear MF with closed-form solutions. Production systems use deep two-tower models where the D-scaling theorem holds only approximately. Netflix's own Section 5 argues the problem is likely *worse* in deep models due to opaque implicit scaling across layers — TurboQuant's formal guarantees provide a hedge against this opacity.
- **Limitation**: Expectation-level unbiasedness is over quantizer randomness. A single instantiated quantizer (one seed) may produce biased estimates for individual pairs. Practical deployment should use averaged or rotated quantizers.
- **Lead into Paper 2**: TurboQuant's data-oblivious design (no codebook retraining) enables real-time incremental item addition — a separate contribution explored in follow-up work.
- **Reproducibility**: Code at https://github.com/2x11-xyz/turboquant-netflix-paper1

---

## References
1. Netflix: arXiv:2403.05440 — Is Cosine-Similarity of Embeddings Really About Similarity?
2. TurboQuant: arXiv:2504.19874 — Online Vector Quantization with Near-optimal Distortion Rate
3. QJL: arXiv:2406.03482 — QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead
4. Product Quantization: Jégou et al. (2011) — Product quantization for nearest neighbor search
5. RaBitQ: Gao & Long (2024) — RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
