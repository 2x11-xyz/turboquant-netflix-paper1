# Paper 1: Preserving Scale-Invariant Scores Under Compression
**Working Title**: Preserving Scale-Invariant User-Item Scores via Near-Optimal Quantization  
**Target**: RecSys Workshop / NeurIPS Efficient ML Workshop (Short Paper, 4 pages max)

---

## 1. Introduction (0.5 page)
- Netflix (2403.05440) proved cosine similarity of learned embeddings is **arbitrary**: equivalent models under diagonal rescaling D produce different cosine values, so cosine-based comparisons are meaningless
- Their fix: use raw dot products, which are provably D-invariant. **This is theoretically correct and settled.**
- **The production gap**: Raw dot products solve the invariance problem, but serving them at scale requires compression. Systems with billions of items cannot store full-precision embeddings — they need compressed representations for memory and ANN retrieval (FAISS, ScaNN, etc.)
- **The compression dilemma**: Standard compression methods (scalar quantization, product quantization) do not guarantee unbiased dot product estimation. Naive MSE-optimal quantization introduces systematic bias — compressed scores systematically underestimate true scores.
- **Our contribution**: We observe that TurboQuant (2504.19874), a near-optimal multi-bit unbiased inner product quantizer, preserves Netflix's D-invariant scores in expectation at 2–3 bits/value. We make this connection explicit, verify it experimentally on Netflix's own synthetic setup, and derive a prescriptive link between Eq.2 weight decay and quantization variance control. Our theoretical observation is a corollary of Netflix Eq.3 and TurboQuant Theorem 2; the contribution is making the connection explicit and demonstrating its practical implications.
- **Architecture note**: TurboQuant is one-sided — items are quantized, user queries remain full-precision. This matches the asymmetric serving pattern in production recommender systems (massive item index, sparse live queries).

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
- **User-item dot product is invariant**: $\langle u^{(D)}, v^{(D)} \rangle = \langle uD, vD^{-1} \rangle = \langle u, v \rangle$ for all D
- Note: item-item dot products $\langle v_i D^{-1}, v_j D^{-1} \rangle = v_i^\top D^{-2} v_j$ are NOT D-invariant
- Netflix's recommendation is correct — but silent on how to **compress** these dot products for production serving

### 2.3 Why Compression Matters
- Full-precision storage is expensive at billion-item scale
- Standard MSE-optimal quantization introduces systematic bias in dot product estimation (we demonstrate this empirically)
- Product Quantization trains data-dependent codebooks that require retraining when item catalogs change
- **Result**: The correct similarity measure (dot product) needs a compression method with formal unbiasedness guarantees

---

## 3. TurboQuant Preserves D-Invariance Under Compression (1 page)
### 3.1 One-Sided Unbiased Estimator (TurboQuant Theorem 2)
TurboQuant_IP quantizes **one argument** (items) while leaving the other (users) in full precision. For unit-norm $x \in S^{d-1}$ (with norms stored separately):
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

Note: $\kappa(D)$ is a worst-case bound. Actual variance depends on the specific alignment of $u,v$ with D's eigenvectors, i.e., $\|uD\|^2 \cdot \|vD^{-1}\|^2$ rather than just $\kappa(D)^2$.

This creates a **prescriptive connection** to Netflix's Eq.2 regularization:
- **Eq.2 controls both factors**: the $\|XA\|_F^2$ term bounds user norms $\|u\|^2$, and the $\|B\|_F^2$ term bounds item norms $\|v\|^2$
- This is strictly stronger than Eq.1, which leaves both norms unconstrained under D-rescaling
- If compression variance is a first-order concern, Eq.2-style norm control provides a cleaner variance regime than Eq.1's rescaling ambiguity
- By AM-GM, penalizing the sum of squared norms ($\|u\|^2 + \|v\|^2$) bounds their product ($\|u\|^2 \cdot \|v\|^2$), which is exactly the variance pre-factor

### 3.3 MSE Preservation (TurboQuant Theorem 1)
$$\mathbb{E}[\|x - Q_{\text{mse}}(x)\|^2] \leq \frac{\pi\sqrt{3}}{2} \cdot 4^{-b}$$
Unlike cosine normalization, TurboQuant does not explicitly discard norm information; its MSE guarantee provides controlled reconstruction error of the original vector.

---

## 4. Experiments (1 page)

### 4.1 Synthetic Data (Netflix Section 4 Setup)
Following Netflix's synthetic setup:
- n=20,000 users, p=1,000 items, C=5 clusters with known ground truth
- Power-law item popularity ($\beta_{\text{item}} \in [0.25, 1.5]$) and user activity ($\beta_{\text{user}} = 0.5$)
- Dirichlet(1,...,1) user-cluster preferences
- Train with Eq.1 ($\lambda=10{,}000$) and Eq.2 ($\lambda=100$), $k=50$
- D-scalings derived from SVD of learned B (matching Netflix Figure 1 methodology)
- Our implementation uses clean-room TurboQuant code from the paper (we identified a scaling bug in the reference library; see Appendix)

### 4.2 Figure 1: Netflix Replication (1 row, 6 panels)
- True item clusters + item-item cosine similarity under 4 D-scalings + Eq.2 reference
- Replicates Netflix's Figure 1: cosine similarity is arbitrary — block structure changes or disappears depending on the choice of D
- This establishes the problem; our contribution follows in the subsequent figures/table

### 4.3 Figure 2: TurboQuant vs MSE-Only Baseline

**Figure 2A — Scatter plots (2 rows × 3 columns):**
- Each dot = one (user, item) pair; y-axis = MC mean estimate over 100 seeds; x-axis = true ⟨u, v⟩
- Top row: TurboQuant 3-bit estimates at κ(D) ∈ {1, 18, 311}
- Bottom row: MSE-only baseline (same quantizer, no QJL residual correction) at same κ values
- **Key observation**: TQ points cluster tightly on y=x at low κ and spread symmetrically with increasing κ (unbiased, growing variance). MSE-only points are systematically below y=x at all κ values (biased, scores underestimated).

**Figure 2B — Bias and variance vs κ(D):**
- Left panel: Mean bias with 95% CI for TQ and MSE-only at 2-bit and 3-bit
- Right panel: Mean variance (log scale) vs κ(D) (log scale)
- **Key observations**:
  - TQ bias stays near zero across all κ ≤ 311 (CIs overlap zero)
  - MSE-only has persistent negative bias (~−0.004 at 3-bit, ~−0.014 at 2-bit)
  - Variance grows with κ for both methods; 3-bit has ~4× lower variance than 2-bit (matching the 4^{−b} bound)
  - At extreme κ (~10⁵), both methods have variance so large that finite MC estimates are unreliable

### 4.4 Table 1: Quantitative Results
D-scaling via $D = \text{diag}(e^{tz_1}, \ldots, e^{tz_k})$, $t \in \{0, 0.5, 1, 2\}$, 200 users × 50 items = 10,000 pairs, 100 MC seeds.

| κ(D) | True ⟨u,v⟩ | TQ 3-bit Mean ± SEM | TQ Bias | TQ Var | MSE-only Bias | MSE-only Var |
|------|-----------|---------------------|---------|--------|---------------|-------------|
| 1 | 0.0377 | 0.0376 ± 0.00002 | −0.0001 | 3.2e-4 | −0.0043 | 1.9e-4 |
| 18 | 0.0377 | 0.0376 ± 0.00005 | −0.0001 | 1.7e-3 | −0.0043 | 9.9e-4 |
| 311 | 0.0377 | 0.0382 ± 0.0003 | +0.0005 | 9.4e-2 | −0.0049 | 4.9e-2 |
| 97K | 0.0377 | 0.187 ± 0.048 | +0.149* | 2.9e+3 | −0.691 | 1.8e+3 |

*At κ=97K, the apparent "bias" is Monte Carlo noise from enormous variance (SEM ≈ 0.048), not genuine bias.

Key findings:
1. True dot product is constant across all κ — D-invariant ✓
2. TQ bias is statistically indistinguishable from zero for κ ≤ 311 ✓
3. MSE-only has persistent, significant negative bias at every κ ✗
4. Variance grows with κ as predicted by the bound ✓
5. 3-bit variance is ~4× lower than 2-bit, matching 4^{−b} scaling ✓

---

## 5. Discussion & Next Steps (0.25 page)
- **Contribution**: Netflix identified the correct similarity measure (dot product); we show TurboQuant provides a near-optimal compressed estimator that preserves it in expectation, and we characterize its variance under rescaling. The theoretical observation is a corollary of two existing results; the contribution is making the connection explicit, providing the prescriptive Eq.2 link, and validating experimentally with an MSE-only ablation.
- **Practical guidance**: If compression variance is a first-order concern, prefer Eq.2-style weight decay — it bounds the norm product that drives quantization variance.
- **Limitation**: Our theory is for linear MF with closed-form solutions. Production systems use deep two-tower models where D-scaling holds only approximately. Netflix's Section 5 argues the problem is likely *worse* in deep models — TurboQuant's formal guarantees provide a hedge against this opacity.
- **Limitation**: Unbiasedness is in expectation over quantizer randomness. A single instantiated quantizer (one seed) may produce biased estimates for individual pairs.
- **Limitation**: One-sided compression (items quantized, users full-precision) matches asymmetric serving but does not address symmetric item-item retrieval.
- **Limitation**: $d=50$ is lower than production dimensions (128–512). TQ variance scales as $1/d$, so results improve at production scale.
- **Implementation note**: We identified a scaling bug in the reference TurboQuant library (`back2matching/turboquant`) where the QJL dequantization coefficient is incorrect. Our experiments use a clean-room implementation verified against the paper's theorems. Bug report filed.
- **Lead into Paper 2**: TurboQuant's data-oblivious design (no codebook retraining) enables real-time incremental item addition — a separate contribution explored in follow-up work.
- **Reproducibility**: Code at https://github.com/2x11-xyz/turboquant-netflix-paper1

---

## References
1. Netflix: arXiv:2403.05440 — Is Cosine-Similarity of Embeddings Really About Similarity?
2. TurboQuant: arXiv:2504.19874 — Online Vector Quantization with Near-optimal Distortion Rate
3. QJL: arXiv:2406.03482 — QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead
4. Product Quantization: Jégou et al. (2011) — Product quantization for nearest neighbor search
5. RaBitQ: Gao & Long (2024) — RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
