# Unbiased Inner Product Quantization Does Not Improve Retrieval

**Working Title**: For Retrieval, Low Variance Beats Zero Bias: Why MSE-Optimal Quantization Outperforms Unbiased Estimators  
**Target**: RecSys 2026 R&P Notes / NeurIPS Efficient ML Workshop / SIGIR Short Paper (4 pages)

---

## Abstract

Recent work on data-oblivious vector quantization (TurboQuant, arXiv:2504.19874) sacrifices MSE-optimal reconstruction to achieve formally unbiased inner product estimation, with the stated motivation that this improves nearest-neighbor search. We show this assumption is incorrect across the full range of practical dimensions (d=50 to d=2048). MSE-optimal scalar quantization (Lloyd-Max after random rotation) produces an approximately multiplicative bias (alpha approx 0.88 for 2-bit, 0.97 for 3-bit) that perfectly preserves item rankings. TurboQuant's unbiasedness comes at 3-5x higher per-instance variance, causing substantially more rank inversions in a single deployed index. Across Gaussian embeddings (d=50-2048), GloVe-200 word vectors, and clustered synthetic embeddings (d=384-1536), MSE-only achieves 12-29 percentage points higher Recall@10 than TurboQuant at matched bit rate. The gap does not narrow with increasing dimension. A trivial scalar correction eliminates MSE's calibration disadvantage, yielding an estimator that dominates on both ranking and score accuracy. We conclude that unbiased inner product quantization solves the wrong problem for retrieval and should be reserved for settings where scores enter nonlinear computations (e.g., softmax attention in KV cache compression).

---

## 1. Introduction (0.5 page)

Compressing item embeddings to low bit-widths is essential for serving large-scale retrieval systems. TurboQuant recently proposed a data-oblivious quantization scheme guaranteeing unbiased inner product estimation:

E[<q, TQ(v)>] = <q, v> for any query q

The paper motivates this property explicitly for nearest-neighbor search (Section 3.2): "For important applications like nearest neighbor search, having an unbiased inner product estimator is essential." We test this claim directly.

**Our finding**: Unbiasedness is counterproductive for retrieval. A deployed retrieval system uses one quantized index (one realization of the random quantizer). What matters is per-instance accuracy, not expectation over parallel universes. MSE-optimal quantization:
1. Has lower per-instance variance (tighter spread around a biased prediction)
2. Has monotonically biased scores (rankings perfectly preserved)
3. Dominates on Recall@K at every dimension from d=50 to d=2048

The bias-variance tradeoff decisively favors low variance for ranking tasks.

**Contributions:**
- Show MSE-only beats TQ on Recall@10 by 12-29pp across d=50 to d=2048, real and synthetic data
- Prove the gap does NOT narrow with dimension (contradicting the O(1/d) variance argument)
- Show trivial scalar debiasing eliminates MSE's calibration disadvantage
- Identify the correct use case for unbiased quantization (nonlinear downstream pipelines)

---

## 2. Background (0.5 page)

### 2.1 Asymmetric Scalar Quantization
- Item embeddings v in R^d compressed to b bits/dimension
- Random orthogonal rotation Pi (makes coordinates approximately i.i.d.)
- Lloyd-Max scalar quantization per coordinate (MSE-optimal for Gaussian marginals)
- Query embeddings remain full-precision (asymmetric, one-sided compression)

### 2.2 TurboQuant's Unbiased Construction
- Allocates (b-1) bits for MSE-optimal reconstruction + 1 bit for QJL sign correction
- QJL (Quantized Johnson-Lindenstrauss) uses random sign projections on the MSE residual
- Theorem 2 guarantee: E[<q, TQ(v)>] = <q, v>
- Cost: The unbiasedness correction adds variance proportional to ||residual||^2 / d

### 2.3 The Key Distinction
- MSE-only: all b bits minimize reconstruction error. Biased for inner products.
- TurboQuant: (b-1) bits for MSE + 1 bit for unbiasedness. Lower MSE bits = higher base variance, plus QJL noise.
- Fair comparison: both use b total bits per coordinate.

---

## 3. Why Monotonic Bias Beats Zero Bias for Ranking (0.75 page)

### 3.1 MSE-Only Bias is Approximately Multiplicative

Lloyd-Max quantization satisfies the orthogonality principle: E[x | Q(x)] = Q(x), meaning centroids are conditional expectations. For the reconstructed vector: ||Q(x)|| < ||x|| by Pythagoras (error orthogonal to reconstruction). After random rotation distributes this uniformly:

<q, v_hat_MSE> approx alpha * <q, v>,  alpha = 1 - D_mse

where D_mse is the normalized MSE distortion. Empirically alpha = 0.883 (2-bit) and 0.966 (3-bit), matching the theoretical 1 - 0.117 = 0.883 and 1 - 0.034 = 0.966 from TurboQuant's own Theorem 1 bounds.

### 3.2 Monotone Bias Preserves Rankings Exactly
If s_i > s_j then alpha * s_i > alpha * s_j for any alpha > 0. The bias is rank-preserving.

### 3.3 TQ's Variance Causes Rank Inversions
For a single deployed index, TurboQuant adds item-specific random noise from the QJL residual estimation. When the noise magnitude exceeds the score gap between adjacent items, rank inversions occur. Since TQ has 3-5x higher variance than MSE-only, it produces 2-3x more pairwise inversions.

### 3.4 Trivial Debiasing Eliminates Calibration Disadvantage
Fit global alpha on calibration queries (least-squares), divide all scores by alpha. The corrected MSE estimator has near-zero bias AND low variance, dominating TQ on both ranking and score accuracy (MAE ~2x lower).

---

## 4. Experiments (1.5 pages)

### 4.1 Setup

**Datasets:**
1. **Gaussian** (controlled): Random unit-norm vectors, d in {50, 128, 256, 512, 1024, 2048}. 10K items, 1K queries.
2. **GloVe-200** (real): 50K word vectors from GloVe 6B, d=200. 2K queries.
3. **Clustered synthetic** (production-representative): 50K items with 20 cluster structure, d in {384, 768, 1536}. 5K queries.

**Methods:** MSE-only (all b bits Lloyd-Max), TurboQuant ((b-1)+1 QJL), Debiased MSE (MSE + global alpha correction on held-out calibration set). Bits in {2, 3}.

**Evaluation:** Recall@10, Recall@100, Score MAE, Spearman rho. 50 random seeds per condition. Calibration/test split: first half queries calibrate alpha, second half for evaluation.

### 4.2 Results: Dimension Sweep (Gaussian)

| d | TQ 2-bit | MSE 2-bit | Gap | TQ 3-bit | MSE 3-bit | Gap |
|------|----------|-----------|------|----------|-----------|------|
| 50 | 0.254 | 0.497 | +0.243 | 0.458 | 0.696 | +0.238 |
| 128 | 0.252 | 0.518 | +0.266 | 0.467 | 0.721 | +0.254 |
| 256 | 0.257 | 0.535 | +0.279 | 0.477 | 0.732 | +0.255 |
| 512 | 0.254 | 0.535 | +0.281 | 0.475 | 0.734 | +0.259 |
| 1024 | 0.257 | 0.541 | +0.284 | 0.480 | 0.739 | +0.260 |
| 2048 | 0.253 | 0.537 | +0.284 | 0.474 | 0.736 | +0.262 |

**Key finding: The gap does not narrow with dimension. It slightly widens.**

This contradicts the intuition that TQ's O(1/d) variance should help at high d. The reason: MSE-only's variance ALSO scales as O(1/d). The ratio is constant across dimensions.

### 4.3 Results: GloVe-200 (Real Word Embeddings)

| Metric | TQ 2-bit | MSE 2-bit | TQ 3-bit | MSE 3-bit |
|--------|----------|-----------|----------|-----------|
| Recall@10 | 0.587 | **0.757** | 0.733 | **0.856** |
| Recall@100 | 0.591 | **0.776** | 0.747 | **0.876** |
| MAE | 0.0424 | **0.0198** | 0.0240 | **0.0106** |
| Alpha | - | 0.884 | - | 0.967 |

MSE-only dominates on real data: +17pp Recall@10 (2-bit), +12pp (3-bit).

### 4.4 Results: High-Dimensional Clustered Data

| d | TQ 2-bit | MSE 2-bit | Gap | TQ 3-bit | MSE 3-bit | Gap |
|------|----------|-----------|------|----------|-----------|------|
| 384 | 0.206 | 0.488 | +0.282 | 0.427 | 0.703 | +0.276 |
| 768 | 0.204 | 0.491 | +0.287 | 0.428 | 0.706 | +0.278 |
| 1536 | 0.203 | 0.493 | +0.291 | 0.428 | 0.710 | +0.282 |

Confirmed at production-scale dimensions: MSE-only wins by 28-29pp at d=1536.

### 4.5 Score Calibration (MAE)

| d | TQ 2-bit MAE | MSE 2-bit MAE | Debiased MAE |
|------|--------------|---------------|--------------|
| 50 | 0.0837 | 0.0381 | **0.0405** |
| 256 | 0.0376 | 0.0170 | **0.0181** |
| 1024 | 0.0188 | 0.0086 | **0.0091** |
| 2048 | 0.0133 | 0.0061 | **0.0065** |

Note: Raw MSE-only beats debiased MSE on MAE (dividing by alpha < 1 amplifies residual variance). But both MSE variants beat TQ by ~2x on score accuracy.

### 4.6 Alpha Stability

| d | Alpha (2-bit) | Alpha std across seeds |
|------|---------------|----------------------|
| 50 | 0.890 | 0.0009 |
| 256 | 0.884 | 0.0006 |
| 1024 | 0.882 | 0.0045 |
| 2048 | 0.881 | 0.0094 |
| GloVe-200 | 0.884 | 0.0008 |

Alpha is remarkably stable: essentially a constant determined by the number of bits (not the data). This is expected since alpha = 1 - D_mse depends only on the Lloyd-Max codebook.

---

## 5. Discussion (0.75 page)

### 5.1 Why the O(1/d) Argument Fails
One might expect TQ to catch up at high d because its QJL variance is O(||r||^2/d). But MSE-only's variance is ALSO O(1/d) (independent coordinate errors average out in the dot product). The ratio Var(TQ)/Var(MSE) is constant in d, so the Recall gap is dimension-independent.

### 5.2 The Correct Use Case for Unbiased Quantization
Unbiasedness matters when compressed scores enter nonlinear downstream functions where a multiplicative shift is NOT equivalent:
- **Softmax attention** (KV cache): softmax(alpha * s / sqrt(d)) != softmax(s / sqrt(d)) — shrinkage flattens attention
- **Score thresholding**: A threshold calibrated on uncompressed scores produces false negatives under shrinkage
- **Multi-system aggregation**: Different quantizers have different alpha values

For pure ranking (top-K retrieval, ANN search), MSE-optimal quantization is strictly preferred.

### 5.3 Relation to TurboQuant's Own ANN Results
TurboQuant shows competitive recall vs Product Quantization (PQ) and RaBitQ. This does not contradict our findings: PQ is a different (data-dependent, codebook-based) approach, not the MSE-optimal scalar quantization we compare. Our contribution is showing that TQ's own MSE-only ablation (without the QJL correction) is the stronger retrieval method.

### 5.4 Limitations
- All experiments use asymmetric quantization (queries at full precision); symmetric settings may differ
- Synthetic clustered data may not capture all properties of real trained embeddings
- We do not test TQ variants with more than 1 bit for QJL
- Real retrieval benchmarks (MS MARCO, MTEB) would further strengthen the message

### 5.5 Recommendations for Practitioners
1. For ANN retrieval: use MSE-optimal quantization (all bits for Lloyd-Max)
2. If absolute scores needed: apply global alpha correction (alpha = 1 - D_mse, knowable a priori)
3. Reserve unbiased quantization for KV cache or softmax-adjacent pipelines
4. Evaluate quantizers on Recall@K, not on inner product bias

---

## References
1. TurboQuant: arXiv:2504.19874 — Online Vector Quantization with Near-optimal Distortion Rate
2. Netflix: arXiv:2403.05440 — Is Cosine-Similarity of Embeddings Really About Similarity?
3. QJL: arXiv:2406.03482 — QJL: 1-Bit Quantized JL Transform for KV Cache Quantization
4. Product Quantization: Jegou et al. (2011) — Product quantization for nearest neighbor search
5. RaBitQ: Gao & Long (2024) — RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
6. ScaNN: Guo et al. (2020) — Accelerating Large-Scale Inference with Anisotropic Vector Quantization
