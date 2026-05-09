# For Retrieval, Low Variance Beats Zero Bias: Why MSE-Optimal Quantization Outperforms Unbiased Estimators

**Target**: NeurIPS 2026 Efficient ML Workshop / RecSys 2026 R&P Notes (4 pages)

---

## Abstract

Data-oblivious vector quantization methods like TurboQuant (arXiv:2504.19874) sacrifice MSE-optimal reconstruction to guarantee unbiased inner product estimation, motivated by the claim that unbiasedness is "essential" for nearest-neighbor search. We show this is incorrect. MSE-optimal scalar quantization (Lloyd-Max after random rotation) produces a global multiplicative shrinkage alpha = 1 - D_mse that perfectly preserves item rankings; the dominant score error is low-variance residual noise. TurboQuant's unbiasedness guarantee adds 2-6x higher per-instance variance, causing substantially more rank inversions in any single deployed index. Across random Gaussian vectors (d=50-2048), GloVe-200 word embeddings, and clustered synthetic data (d=384-1536), MSE-only achieves 1.4-29 percentage points higher Recall@10 at matched bit rate, with the gap persisting from 2-bit through 8-bit quantization and never narrowing with dimension. We conclude that for asymmetric top-K retrieval, practitioners should allocate all bits to MSE-optimal reconstruction and reserve unbiased inner product quantization for pipelines where scores enter nonlinear computations (e.g., softmax attention).

---

## 1. Introduction (0.5 page)

TurboQuant proposes data-oblivious vector quantization with a formal guarantee: E[<q, TQ(v)>] = <q, v>. Their motivation (Section 3.2): "For important applications like nearest neighbor search, having an unbiased inner product estimator is essential."

We test this claim directly by comparing TurboQuant against its own MSE-only ablation — the same architecture with all bits allocated to Lloyd-Max reconstruction, without the QJL unbiasedness correction. This is the fairest possible comparison: same random rotation, same computational cost, same bit budget.

**Our finding**: The unbiasedness guarantee is counterproductive for retrieval. A deployed system uses one quantized index. Per-instance accuracy, not expectation over randomness, determines ranking quality. MSE-optimal quantization wins because:

1. Its bias is a predictable global multiplicative shrinkage (alpha approx 0.88-0.99 depending on bit-width) that preserves all pairwise orderings
2. Its per-instance variance is 2-6x lower than TurboQuant's, producing far fewer rank inversions
3. The shrinkage factor alpha = 1 - D_mse follows directly from TurboQuant's own Theorem 1 bounds

**Contributions:**
- Prove that MSE-only inner product shrinkage equals 1 - D_mse (TQ's own distortion bound)
- Show MSE-only dominates TQ on Recall@10 across d=50-2048, bits=2-8, real and synthetic data
- Demonstrate the gap is dimension-independent (contradicting O(1/d) variance intuition)
- Identify the correct use case for unbiased quantization (nonlinear downstream pipelines)

---

## 2. Background (0.5 page)

### 2.1 Asymmetric Scalar Quantization
- Item embeddings v in R^d compressed to b bits/dimension; queries at full precision
- Random orthogonal rotation Pi (makes coordinates approximately i.i.d. Gaussian)
- Lloyd-Max scalar quantization per coordinate (MSE-optimal for Gaussian marginals)
- Norm stored separately (32 bits); reconstruction: v_hat = ||v|| * Pi * Q(Pi^T v / ||v||)

### 2.2 TurboQuant's Unbiased Construction (Algorithm 2)
- Allocates (b-1) bits for MSE-optimal quantization + 1 bit for QJL sign correction on residual
- QJL correction: stores sign(S * residual) where S is random Gaussian; dequantizes via S^T * signs * ||residual|| * sqrt(pi/2)/d
- Guarantee: E[<q, TQ(v)>] = <q, v> (Theorem 2)
- Storage: b*d bits + 32 bits (norm) + 32 bits (residual norm) per vector

### 2.3 Fair Comparison
Both methods use b total bits per coordinate. TQ additionally stores a 32-bit residual norm scalar — at d >= 200 this is < 1% overhead (negligible); at d=50 it is ~4% overhead. We compare at matched b*d bits and note TQ's slight storage advantage is insufficient to explain the observed gaps.

---

## 3. MSE-Only Bias is Predictable, Monotone, and Harmless (0.75 page)

### 3.1 Proposition: Inner Product Shrinkage = 1 - D_mse

**Proposition.** For unit-norm vectors quantized by Lloyd-Max after random rotation:
E[<q, v_hat>] = (1 - D_mse) * <q, v>

*Proof sketch.* Lloyd-Max orthogonality: E[(X - Q(X)) * Q(X)] = 0 per coordinate, hence E[X * Q(X)] = E[Q(X)^2]. For rotated coordinate X_j of the unit vector: E[X_j * Q(X_j)] = E[X_j^2] - D_mse * E[X_j^2] = (1 - D_mse)/d. Summing over d coordinates and applying to the inner product via rotation invariance gives the result.

**Empirical verification (from TurboQuant's own Theorem 1 constants):**

| Bits | D_mse (TQ Thm 1) | Predicted alpha = 1 - D_mse | Measured alpha |
|------|-------------------|----------------------------|----------------|
| 2 | 0.117 | 0.883 | 0.883 |
| 3 | 0.034 | 0.966 | 0.966 |
| 4 | 0.009 | 0.991 | 0.992 |
| 8 | ~0.0001 | ~0.9999 | 0.9997 |

The match is near-perfect. Our empirical shrinkage IS the TurboQuant paper's own distortion constant.

### 3.2 Monotone Shrinkage Preserves Rankings
Since alpha > 0 for any b >= 1: s_i > s_j implies alpha * s_i > alpha * s_j. The bias component is perfectly rank-preserving. Ranking degradation comes entirely from residual noise (which MSE-only minimizes by construction).

### 3.3 TQ's Variance is 2-6x Higher
The QJL correction adds variance proportional to ||residual||^2 * (pi/2) / d. The variance ratio:

Var(TQ) / Var(MSE) approx (pi/2) * D_mse(b-1) / D_mse(b)

At 2-bit: (pi/2) * 0.36 / 0.117 approx 4.8x. At 4-bit: (pi/2) * 0.034 / 0.009 approx 5.9x. Empirically confirmed by the MAE ratios (2.0-2.4x on absolute error, implying 4-6x on variance).

### 3.4 Both Variances Scale as O(1/d) — Ratio is Constant
MSE-only score variance: Var_MSE ~ D_mse * ||q||^2 * ||v||^2 / d
TQ score variance: Var_TQ ~ (pi/2) * D_mse(b-1) * ||q||^2 * ||v||^2 / d

Both scale identically with dimension. The ratio Var_TQ/Var_MSE is independent of d, which is why the Recall gap never narrows.

---

## 4. Experiments (1.5 pages)

### 4.1 Setup

**Datasets:**
1. **Gaussian** (controlled): Random unit-norm vectors, d in {50, 128, 256, 512, 1024, 2048}. 10K items, 1K queries.
2. **GloVe-200** (real): 50K word vectors from GloVe 6B, d=200. 2K queries.
3. **Clustered synthetic** (production-representative): 50K items, 20 clusters, d in {384, 768, 1536}. 5K queries.

**Methods:** MSE-only (all b bits Lloyd-Max), TurboQuant ((b-1)+1 QJL). Bits in {2, 3, 4, 8}.

**Evaluation:** Recall@10, Recall@100, Score MAE. 50 random seeds per condition. First half of queries calibrates alpha; second half for evaluation.

**Implementation:** Clean-room TurboQuant verified against Theorem 2 (bias < 10^{-6} across all conditions). Code will be released.

### 4.2 Main Result: MSE Dominates at All Bit-Widths

**Table 1: Recall@10 gap (MSE minus TQ) by bit-width and dimension**

| d | 2-bit | 3-bit | 4-bit | 8-bit |
|------|---------|---------|---------|---------|
| 50 | +0.243 | +0.238 | +0.165 | +0.014 |
| 256 | +0.279 | +0.255 | +0.169 | +0.017 |
| 512 | +0.281 | +0.259 | +0.171 | +0.018 |
| 1024 | +0.284 | +0.260 | +0.171 | +0.022 |
| 2048 | +0.284 | +0.262 | +0.175 | +0.025 |

MSE-only wins at every operating point. The gap narrows with bit-width (as expected — higher bits means less residual for QJL to corrupt) but NEVER with dimension.

### 4.3 GloVe-200 (Real Data)

| Metric | TQ 2-bit | MSE 2-bit | TQ 3-bit | MSE 3-bit |
|--------|----------|-----------|----------|-----------|
| Recall@10 | 0.587 | **0.757** | 0.733 | **0.856** |
| Recall@100 | 0.591 | **0.776** | 0.747 | **0.876** |
| MAE | 0.042 | **0.020** | 0.024 | **0.011** |

Confirmed on real word embeddings: +17pp (2-bit), +12pp (3-bit).

### 4.4 High-Dimensional Clustered Data

| d | TQ 2-bit | MSE 2-bit | TQ 3-bit | MSE 3-bit |
|------|----------|-----------|----------|-----------|
| 384 | 0.206 | **0.488** | 0.427 | **0.703** |
| 768 | 0.204 | **0.491** | 0.428 | **0.706** |
| 1536 | 0.203 | **0.493** | 0.428 | **0.710** |

At d=1536 (OpenAI embedding scale): MSE wins by +29pp (2-bit), +28pp (3-bit).

### 4.5 Score Accuracy (MAE)

TQ has 2.0-2.4x worse MAE than MSE-only at every operating point, despite having zero bias. This directly demonstrates that unbiasedness does not imply accuracy — variance dominates.

---

## 5. Discussion (0.75 page)

### 5.1 Relation to TurboQuant's Own ANN Experiments
TurboQuant (Section 4.4) shows competitive recall vs Product Quantization (PQ) and RaBitQ on DBpedia/GloVe. This does not contradict our results: their baselines are data-dependent methods (PQ requires k-means codebook training), not MSE-optimal scalar quantization. Our finding is that TurboQuant's own MSE-only ablation — the same data-oblivious architecture without QJL — is the stronger retrieval method at every bit-width.

### 5.2 When Unbiased Quantization IS Appropriate
Unbiasedness matters when scores enter nonlinear functions where multiplicative shrinkage is not equivalent:
- **Softmax attention** (KV cache): softmax(alpha*s/sqrt(d)) != softmax(s/sqrt(d))
- **Score thresholding**: A threshold calibrated on full-precision scores produces false negatives under shrinkage
- TurboQuant's KV cache results (Section 4.3) confirm this: unbiasedness achieves exact quality match at 3.5 bits

For ranking (top-K retrieval), the monotone transformation alpha*s preserves all orderings, making unbiasedness unnecessary.

### 5.3 Limitations
- Asymmetric quantization only (queries at full precision)
- Synthetic clustered data may not capture all properties of trained embeddings
- We use TurboQuant's scalar QJL variant; multi-bit QJL variants may reduce the gap
- Our implementation uses random Gaussian S matrices per the paper; optimized implementations may differ

### 5.4 Recommendations
1. For ANN retrieval: allocate all bits to MSE-optimal reconstruction
2. Alpha is knowable a priori from D_mse — no calibration data needed for score correction
3. Reserve unbiased quantization for KV cache and softmax-adjacent pipelines
4. Evaluate quantizers on Recall@K, not on inner product bias

---

## References
1. TurboQuant: arXiv:2504.19874 — Near-optimal Online Vector Quantization
2. QJL: arXiv:2406.03482 — QJL: 1-Bit Quantized JL Transform
3. Product Quantization: Jegou et al. (2011)
4. RaBitQ: Gao & Long (2024)
5. ScaNN: Guo et al. (2020) — Anisotropic Vector Quantization
6. GloVe: Pennington et al. (2014)
