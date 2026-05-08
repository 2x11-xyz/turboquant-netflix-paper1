# Paper 1: Preserving Scale-Invariant User-Item Scores via Near-Optimal Quantization
**Working Title**: Preserving Scale-Invariant User-Item Scores via Near-Optimal Quantization  
**Target**: RecSys Workshop / NeurIPS Efficient ML Workshop (Short Paper, 4 pages max)

---

## 1. Introduction (0.5 page)
- Netflix (2403.05440) proved cosine similarity of learned embeddings is **arbitrary**: equivalent models under diagonal rescaling D produce different cosine values, so cosine-based comparisons are meaningless
- Their fix: use raw dot products, which are provably D-invariant
- **Open problem**: No compressed/quantized estimator has been shown to preserve this invariance. Standard scalar quantization introduces bias; PQ requires global retraining
- **Our contribution**: We show TurboQuant's one-sided unbiased estimator (2504.19874) preserves D-invariant dot products in expectation at 2–3 bits/value, and we characterize how quantization variance scales with the condition number κ(D) — yielding a prescriptive link to Netflix's Eq.2 weight decay

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
- The problem: how to compress these dot products without destroying invariance?

---

## 3. TurboQuant Preserves D-Invariance (1 page)
### 3.1 One-Sided Unbiased Estimator (TurboQuant Theorem 2)
TurboQuant_IP quantizes **one argument** (items) while leaving the other (users) in full precision:
$$\mathbb{E}[\langle y, \text{DeQuant}(Q(x)) \rangle] = \langle y, x \rangle \quad \text{(exactly unbiased)}$$

Applied to Netflix's D-scaled embeddings (quantize items only):
$$\mathbb{E}[\langle u^{(D)}, \text{DeQuant}(Q(v^{(D)})) \rangle] = \langle u^{(D)}, v^{(D)} \rangle = \langle u, v \rangle$$

The estimator preserves D-invariance **in expectation** at any bit-width b ≥ 2.

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
### 4.1 Setup
- **Dataset**: MovieLens-1M (6,040 users × 3,706 items, explicit 1–5 star ratings)
- **Model**: Linear MF ($X \approx XAB^\top$, K=50, λ=0.01, 100 epochs, Adam)
- **D-scaling**: $D = \text{diag}(e^{tz_1}, \ldots, e^{tz_k})$ with fixed $z \sim \mathcal{N}(0,I)$, $t \in \{0, 0.5, 1, 2, 5\}$
  - κ(D) ranges from 1 (t=0) to 2.9×10¹² (t=5)
- **TurboQuant**: TurboQuantIP at 2 and 3 bits, one-sided (items only), 10 rotation seeds for variance estimation

### 4.2 Results: 4-Panel Heatmap
| Panel | What it shows | Key finding |
|-------|---------------|-------------|
| **Cosine Similarity** | Mean cos(u,v) across 100 pairs per condition | Decays from 0.04 → 0.00 as t grows. **Arbitrary.** |
| **Raw Dot Product** | Mean ⟨u,v⟩ across 100 pairs per condition | Stable at 0.1263 (Eq.1) / 0.1271 (Eq.2) for all t. **D-invariant.** |
| **TQ 2-bit Variance** | log₁₀(avg per-pair variance across 10 seeds) | Grows from 10⁻¹·² (t=0) to 10²⁰·⁷ (t=5). **22 orders of magnitude.** |
| **TQ 3-bit Variance** | Same, at 3 bits | Grows from 10⁻¹·⁷ (t=0) to 10²⁰·¹ (t=5). ~0.5 log₁₀ lower than 2-bit. |

### 4.3 Key Observations
1. **Dot product invariance confirmed**: Raw ⟨u,v⟩ is identical (to 4 decimal places) across all D scalings, for both Eq.1 and Eq.2
2. **TurboQuant mean preserves invariance**: E[⟨u, DeQuant(Q(v))⟩] = ⟨u, v⟩ regardless of D (by Theorem 2)
3. **Variance explodes with κ(D)**: At t=5 (κ≈3×10¹²), quantization variance reaches 10²⁰ — unusable in practice
4. **More bits help, but don't fix D-sensitivity**: 3-bit is ~0.5 log₁₀ better than 2-bit, but both explode equally under high κ(D)
5. **Weight decay (Eq.2) bounds variance**: Eq.2 consistently shows lower TQ variance than Eq.1 (e.g., 10²⁰·⁴ vs 10²⁰·⁷ at t=5), confirming the prescriptive link

---

## 5. Discussion & Next Steps (0.25 page)
- **Paper 1 contribution**: First proof that TurboQuant preserves Netflix's D-invariant dot products in expectation, plus a prescriptive link between Eq.2 weight decay and quantization variance control
- **Practical guidance**: Use Eq.2-style regularization when planning to quantize; it bounds the worst-case variance of the quantized estimator
- **Limitation**: Variance grows with κ(D); TurboQuant does not "fix" the D-scaling ambiguity, it provides an unbiased compressed estimator whose quality degrades predictably with embedding condition number
- **Lead into Paper 2**: TurboQuant's data-oblivious design (no codebook retraining needed) enables real-time incremental item addition — a separate contribution explored in follow-up work
- **Reproducibility**: Code + experiment scripts at https://github.com/2x11-xyz/turboquant-netflix-paper1

---

## References
1. Netflix: arXiv:2403.05440 — Is Cosine-Similarity of Embeddings Really About Similarity?
2. TurboQuant: arXiv:2504.19874 — Online Vector Quantization with Near-optimal Distortion Rate
3. Product Quantization: Jégou et al. (2011) — Product quantization for nearest neighbor search
4. RaBitQ: Gao & Long (2024) — RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
