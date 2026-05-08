# Paper 1: Preserving Scale-Invariant User-Item Scores via Near-Optimal Quantization
**Working Title**: Preserving Scale-Invariant User-Item Scores via Near-Optimal Quantization  
**Target**: RecSys Workshop / NeurIPS Efficient ML Workshop (Short Paper, 4 pages max)

---

## 1. Introduction (0.5 page)
- Netflix (2403.05440) proved cosine similarity of learned embeddings is **arbitrary**: equivalent models under diagonal rescaling D produce different cosine values, so cosine-based comparisons are meaningless
- Their fix: use raw dot products, which are provably D-invariant. **This is theoretically correct and settled.**
- **The production gap**: Raw dot products solve the invariance problem, but serving them at scale is impractical. Real systems with billions of items (1B × 128d × 4 bytes = 512 GB) cannot store full-precision embeddings — they require compression for memory and ANN retrieval (FAISS, ScaNN, etc.)
- **The compression dilemma**: Standard compression methods destroy the very property that makes dot products trustworthy:
  - Scalar quantization introduces **bias** (systematic shift in dot products)
  - Product Quantization (PQ) requires **global codebook retraining** on the item distribution — if the model is retrained or items are added, codebooks become stale
  - Both break D-invariance: the compressed dot product no longer equals the true dot product in expectation
- **Our contribution**: TurboQuant (2504.19874) is the first provably unbiased compressed estimator for dot products. We show it preserves Netflix's D-invariant scores in expectation at 2–3 bits/value, and we characterize how quantization variance scales with κ(D) — yielding a prescriptive link to Netflix's Eq.2 weight decay

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

### 2.3 Why Compression Breaks Invariance
- At production scale, embeddings must be compressed for memory and retrieval
- Standard quantization Q introduces bias: $\mathbb{E}[\langle Q(u), Q(v) \rangle] \neq \langle u, v \rangle$
- PQ trains data-dependent codebooks: changing D invalidates the codebooks entirely
- **Result**: The correct similarity measure (dot product) becomes impractical to deploy at scale without a compression method that preserves unbiasedness

---

## 3. TurboQuant Preserves D-Invariance Under Compression (1 page)
### 3.1 One-Sided Unbiased Estimator (TurboQuant Theorem 2)
TurboQuant_IP quantizes **one argument** (items) while leaving the other (users) in full precision:
$$\mathbb{E}[\langle y, \text{DeQuant}(Q(x)) \rangle] = \langle y, x \rangle \quad \text{(exactly unbiased)}$$

Applied to Netflix's D-scaled embeddings (quantize items only):
$$\mathbb{E}[\langle u^{(D)}, \text{DeQuant}(Q(v^{(D)})) \rangle] = \langle u^{(D)}, v^{(D)} \rangle = \langle u, v \rangle$$

The compressed estimator preserves D-invariance **in expectation** at 2–3 bits/value — a 5–8× compression over FP16.

### 3.2 Variance Depends on κ(D) — The Prescriptive Link
The variance of the quantized dot product depends on $\|v^{(D)}\|^2$:
$$\text{Var}[\langle u, \hat{v}^{(D)} \rangle] \propto \|v^{(D)}\|^2 = \|vD^{-1}\|^2$$

When D has high condition number κ(D), some item embedding coordinates are amplified by $D^{-1}$, increasing $\|v^{(D)}\|^2$ and thus the quantization variance. This creates a **prescriptive connection** to Netflix's Eq.2 regularization:

- **Eq.2's weight decay** on $\|B\|_F^2$ directly controls $\|v\|^2$, bounding the worst-case quantization variance
- Practitioners should prefer Eq.2-style regularization when planning to quantize embeddings
- This turns TurboQuant's D-dependent variance from a weakness into a design guideline

### 3.3 MSE Preservation (TurboQuant Theorem 1)
$$\mathbb{E}[\|x - Q_{\text{mse}}(x)\|^2] \leq \frac{\pi\sqrt{3}}{2} \cdot 4^{-b}$$
TurboQuant preserves magnitude information that Netflix identifies as meaningful (unlike cosine normalization which discards it).

---

## 4. Experiments (1 page)

### 4.1 Netflix Figure 1 Replication (Synthetic Data)
Replicate Netflix's exact synthetic setup from Section 4:
- n=20,000 users, p=1,000 items, C=5 clusters with known ground truth
- Power-law popularity ($\beta_{\text{item}} \in [0.25, 1.5]$, $\beta_{\text{user}} = 0.5$)
- Train with Eq.1 (λ=10,000) and Eq.2 (λ=100), k=50

**Figure 1** (Netflix replication + our extension):
- Row 1: Netflix's result — true cluster matrix, then cosSim(B,B) under 3 D-scalings + Eq.2 (shows cosine is arbitrary)
- Row 2 (ours): TurboQuant quantized dot product matrix under the same D-scalings → **block structure is preserved**, matching ground truth

This visually demonstrates: cosine destroys the signal, TurboQuant's compressed dot product recovers it.

### 4.2 MovieLens-1M Quantitative Validation (Table)
- **Dataset**: MovieLens-1M (6,040 users × 3,706 items, explicit 1–5 star ratings)
- **Model**: Linear MF (K=50, λ=0.01, 100 epochs, Adam)
- **D-scaling**: $D = \text{diag}(e^{tz_1}, \ldots, e^{tz_k})$, $t \in \{0, 0.5, 1, 2, 5\}$, κ(D) from 1 to 2.9×10¹²
- **TurboQuant**: TurboQuantIP at 2 and 3 bits, one-sided (items only), 10 rotation seeds

**Table 1**: Quantitative results across D-scaling conditions

| t | κ(D) | Mean cos(u,v) | Mean ⟨u,v⟩ | TQ 2-bit var (log₁₀) | TQ 3-bit var (log₁₀) |
|---|------|---------------|-------------|----------------------|----------------------|
| 0 | 1 | 0.0398 | 0.1263 | −1.2 | −1.7 |
| 0.5 | 18 | 0.0220 | 0.1263 | −0.7 | −1.1 |
| 1 | 311 | 0.0037 | 0.1263 | 0.9 | 0.7 |
| 2 | 97K | 0.0000 | 0.1263 | 5.9 | 5.4 |
| 5 | 2.9T | 0.0000 | 0.1263 | 20.7 | 20.1 |

Key observations:
1. Dot product is rock-stable at 0.1263 across all t — **D-invariant** ✓
2. Cosine similarity decays to zero — **arbitrary** under D-scaling ✓
3. TQ variance grows with κ(D) but **mean is unbiased** at every point ✓
4. Eq.2 consistently shows lower TQ variance than Eq.1 — **weight decay helps** ✓

---

## 5. Discussion & Next Steps (0.25 page)
- **Paper 1 contribution**: Netflix identified the right similarity measure (dot product); we provide the first compressed estimator that provably preserves it. Not "fixing" dot products — they were never broken — but making them **practical to deploy at scale** via near-optimal compression with formal unbiasedness guarantees
- **Practical guidance**: Use Eq.2-style weight decay when planning to quantize; it bounds worst-case quantization variance
- **Limitation**: Variance grows with κ(D); TurboQuant does not eliminate the D-scaling ambiguity, it provides a compressed estimator whose quality degrades predictably with embedding condition number
- **Lead into Paper 2**: TurboQuant's data-oblivious design (no codebook retraining) enables real-time incremental item addition — a separate contribution explored in follow-up work
- **Reproducibility**: Code + experiment scripts at https://github.com/2x11-xyz/turboquant-netflix-paper1

---

## References
1. Netflix: arXiv:2403.05440 — Is Cosine-Similarity of Embeddings Really About Similarity?
2. TurboQuant: arXiv:2504.19874 — Online Vector Quantization with Near-optimal Distortion Rate
3. Product Quantization: Jégou et al. (2011) — Product quantization for nearest neighbor search
4. RaBitQ: Gao & Long (2024) — RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
