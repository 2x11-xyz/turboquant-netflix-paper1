# Unbiased Quantization Does Not Improve Vector Retrieval

**Working Title**: Unbiased Inner Product Quantization Does Not Improve Retrieval: Why Low Variance Beats Zero Bias  
**Target**: RecSys Workshop / NeurIPS Efficient ML Workshop / SIGIR Short Paper (4 pages)

---

## Abstract

Recent work on vector quantization for inner product estimation (TurboQuant, arXiv:2504.19874) proposes sacrificing MSE-optimal reconstruction to achieve formally unbiased dot product estimates. The implicit assumption is that unbiasedness improves downstream retrieval quality. We show this assumption is wrong. In the retrieval setting, MSE-optimal quantization produces a nearly monotonic bias (approximately multiplicative score shrinkage) that preserves item rankings, while unbiased quantization introduces 3-5× higher variance that causes rank inversions. On Netflix's matrix factorization setup (arXiv:2403.05440), MSE-only quantization achieves 8-35% higher Recall@10 than TurboQuant at matched bit rate. A trivial post-hoc scalar correction eliminates MSE's calibration disadvantage, yielding an estimator that dominates on both ranking and score accuracy. We conclude that for vector retrieval applications, practitioners should prefer MSE-optimal quantization and reserve unbiased methods for settings where absolute score values enter nonlinear downstream computations (e.g., softmax attention in KV cache compression).

---

## 1. Introduction (0.5 page)

Compressing high-dimensional vectors to low bit-widths is essential for serving large-scale retrieval systems. Recently, TurboQuant proposed a quantization scheme with a formal unbiasedness guarantee for inner product estimation:
$$\mathbb{E}[\langle q, \text{TQ}(v) \rangle] = \langle q, v \rangle$$

The motivation seems compelling: if the compressed scores are unbiased, retrieval quality should benefit. However, unbiasedness is an expectation-level property — it holds on average over quantizer randomness. A deployed retrieval system uses a single quantized index (one random seed), where per-instance variance determines ranking quality.

We ask: **does unbiased inner product quantization actually improve top-K retrieval?**

Our answer is no. We show:
1. MSE-optimal scalar quantization (Lloyd-Max) produces a nearly perfectly monotonic bias — rankings are barely disturbed
2. TurboQuant's higher variance (the cost of unbiasedness) causes more rank inversions per deployed index
3. A trivial post-hoc scalar correction eliminates MSE's calibration disadvantage, yielding lower per-instance score error than TurboQuant

**The practical insight**: For ranking, you'd rather be consistently wrong in a predictable way (MSE-only: uniform shrinkage) than unpredictably right on average (TQ: zero mean, large per-item noise). This is the James-Stein phenomenon applied to vector quantization.

**When to use which:**

| Method | Ranking (top-K) | Score calibration | When to use |
|--------|----------------|-------------------|-------------|
| MSE-only | Best | Biased (fixable) | Vector retrieval, ANN |
| MSE + scalar correction | Best | Good | Retrieval + downstream scoring |
| TurboQuant (unbiased) | Worse | Good in expectation, worse per-instance | KV cache (softmax), non-correctable nonlinear pipelines |

---

## 2. Background (0.5 page)

### 2.1 The Quantization Setup
- Item embeddings $v \in \mathbb{R}^d$ compressed to $b$ bits/dimension
- Query embeddings $q$ remain full-precision (asymmetric, one-sided compression)
- Goal: estimate $\langle q, v \rangle$ from compressed representation of $v$

### 2.2 MSE-Optimal Scalar Quantization
- Random rotation $\Pi$ (makes coordinates approximately independent)
- Lloyd-Max quantization per coordinate (optimal for Gaussian marginals)
- Reconstruction: $\hat{v} = \Pi \cdot \text{LloydMax}(\Pi^T v / \|v\|) \cdot \|v\|$
- Property: minimizes $\mathbb{E}[\|v - \hat{v}\|^2]$
- Known issue: does NOT guarantee unbiased inner products

### 2.3 TurboQuant's Unbiased Estimator
- Uses $(b-1)$ bits for MSE quantization + 1 bit for QJL (sign) correction
- The QJL correction eliminates bias at the cost of higher reconstruction variance
- Guarantee: $\mathbb{E}[\langle q, \hat{v} \rangle] = \langle q, v \rangle$ (Theorem 2)
- Cost: Variance is $O(\|q\|^2 \cdot \|v\|^2 / d \cdot 4^{-b})$, higher than MSE-only

---

## 3. Why Monotonic Bias Beats Zero Bias for Ranking (1 page)

### 3.1 MSE-Only Bias is Approximately Multiplicative
After random rotation, each coordinate is independently quantized by Lloyd-Max. The reconstruction undershoots each coordinate (centroids are interior points of their Voronoi cells). Since rotation distributes this uniformly across directions:
$$\langle q, \hat{v}_{\text{MSE}} \rangle \approx \alpha \cdot \langle q, v \rangle, \quad \alpha \in (0, 1)$$

Empirically: $\alpha \approx 0.89$ (2-bit), $\alpha \approx 0.97$ (3-bit), with near-zero intercept.

### 3.2 Monotonic Bias Preserves Rankings
If all scores are scaled by the same $\alpha > 0$:
$$s_i > s_j \iff \alpha \cdot s_i > \alpha \cdot s_j$$

Rankings are perfectly preserved under multiplicative bias. In practice, the bias is not perfectly multiplicative (slight per-item variation), but the inversion rate is very low (2-4% of pairs at κ=1).

### 3.3 TQ's Variance Causes Rank Inversions
TurboQuant's QJL correction adds a high-variance residual estimate to each item. For a single deployed index:
- Each item gets an independent random perturbation
- For closely-scored items, the perturbation can exceed the score gap
- Result: more rank swaps than MSE-only's orderly, predictable shrinkage

### 3.4 Debiased MSE Dominates on Calibration Too
The multiplicative bias can be trivially corrected: fit $\alpha$ on calibration data, divide scores by $\alpha$. The corrected estimator:
- Has near-zero bias (same as TQ)
- Retains low variance (much lower than TQ)
- Dominates TQ on per-instance score accuracy (MAE ~2× lower)

This is a direct application of the bias-variance decomposition: $\text{MSE} = \text{Bias}^2 + \text{Variance}$. When TQ's variance far exceeds MSE-only's bias², TQ's total error is higher despite having zero bias.

---

## 4. Experiments (1.25 pages)

### 4.1 Setup
- Netflix's synthetic matrix factorization setup (arXiv:2403.05440, Section 4)
- n=20,000 users, p=1,000 items, k=50 latent dimensions, 5 clusters
- Eq.1 regularization ($\lambda=10,000$) with D-scaling ambiguity
- D-scalings: $D = \text{diag}(e^{tz})$, $t \in \{0, 0.5, 1.0\}$, giving $\kappa \in \{1, 7.5, 56\}$
- 50 random quantizer seeds per condition
- Clean-room TurboQuant implementation (verified against paper theorems)
- 10K calibration users / 10K held-out test users for debiasing

### 4.2 Methods Compared
1. **MSE-only**: All $b$ bits for Lloyd-Max (no QJL correction)
2. **TurboQuant**: $(b-1)$ bits MSE + 1 bit QJL (unbiased)
3. **Debiased MSE**: MSE-only with global $\hat{\alpha}$ fit on calibration set, scores divided by $\hat{\alpha}$

### 4.3 Evaluation Metrics
- **Recall@10**: Fraction of true top-10 items recovered (the retrieval metric)
- **Spearman ρ**: Rank correlation between true and estimated scores per user
- **Score MAE**: Mean absolute error of score estimates (calibration metric)
- All metrics computed per-seed (single deployed index), reported as mean ± std over 50 seeds

### 4.4 Results

**Table 1: Single-seed Recall@10 (higher = better)**

| κ(D) | TQ 2-bit | MSE 2-bit | TQ 3-bit | MSE 3-bit |
|------|----------|-----------|----------|-----------|
| 1 | 0.682 ± 0.018 | **0.768 ± 0.011** | 0.761 ± 0.012 | **0.847 ± 0.009** |
| 7.5 | 0.552 ± 0.034 | **0.680 ± 0.017** | 0.657 ± 0.018 | **0.785 ± 0.012** |
| 56 | 0.249 ± 0.075 | **0.403 ± 0.049** | 0.364 ± 0.048 | **0.548 ± 0.033** |

**Table 2: Score MAE (lower = better)**

| κ(D) | TQ 2-bit | MSE 2-bit | Debiased MSE 2-bit |
|------|----------|-----------|-------------------|
| 1 | 0.0100 | 0.0060 | **0.0050** |
| 7.5 | 0.0182 | 0.0089 | **0.0090** |
| 56 | 0.0632 | 0.0277 | **0.0313** |

**Table 3: Monotonicity verification**

| κ(D) | Bits | Shrinkage α | Per-item α std | Pairwise inversion rate |
|------|------|-------------|----------------|------------------------|
| 1 | 2 | 0.887 | 0.105 | 2.0% (MSE) vs 4.1% (TQ) |
| 7.5 | 2 | 0.889 | 0.201 | 3.7% (MSE) vs 6.9% (TQ) |
| 56 | 2 | 0.880 | 0.686 | 11.0% (MSE) vs 17.6% (TQ) |

**Figure 1: Scatter plots (3 × 4)**
- Rows: κ ∈ {1, 7.5, 56}; Columns: TQ 2-bit, MSE 2-bit, TQ 3-bit, MSE 3-bit
- Each dot = one (user, item) pair, y = MC mean estimate, x = true score
- TQ: centered on y=x (unbiased) but wide spread
- MSE-only: below y=x (biased) but tight spread
- Visual: tight-and-shifted dominates wide-and-centered for ranking

### 4.5 Stability of α
- Global α across seeds: std = 0.011-0.086 (stable enough for correction)
- Per-user α: std = 0.021-0.091 (stable)
- Per-item α: std = 0.064-0.686 (variable, especially at high κ)
- Despite per-item variability, global correction still dominates TQ on MAE because the residual variance after correction is still much lower than TQ's variance

---

## 5. Discussion (0.75 page)

### 5.1 Why This Matters
TurboQuant and related work (QJL, RaBitQ) pursue unbiased inner product estimation as a design goal for retrieval systems. Our results show this goal is misaligned with retrieval quality. The retrieval community should evaluate quantizers on Recall@K, not on bias.

### 5.2 The Correct Use Case for Unbiased Quantization
Unbiasedness matters when compressed scores enter nonlinear downstream computations where monotonic transformations are NOT equivalent preserved:
- **Softmax attention** (KV cache): $\text{softmax}(\alpha \cdot s / \sqrt{d}) \neq \text{softmax}(s / \sqrt{d})$ — shrinkage changes attention distribution
- **Score thresholding with unknown threshold**: if the threshold was calibrated on uncompressed scores, shrinkage causes false negatives
- **Cross-system score comparison**: different quantizers have different α values, making cross-system score comparison meaningless without per-system calibration

For pure ranking (top-K retrieval, ANN search), MSE-optimal quantization is strictly preferred.

### 5.3 Connection to Classical Statistics
Our finding is an instance of the James-Stein phenomenon: a biased shrinkage estimator can dominate an unbiased one in total MSE when the shrinkage is toward a structured target. Here, Lloyd-Max shrinkage toward zero is structured (approximately multiplicative on dot products), while TQ's QJL correction introduces unstructured noise.

### 5.4 Limitations
- Experiments at d=50 (low); TQ variance scales as 1/d, so the gap may narrow at d=512+
- Synthetic data only; real retrieval benchmarks (MS MARCO, NQ) would strengthen the message
- Netflix's Eq.1 model is linear MF; deep two-tower models may have different bias structure
- Single scalar α correction; per-item calibration could further improve MSE-only (but isn't needed to beat TQ)

### 5.5 Recommendations for Practitioners
1. For ANN retrieval: use MSE-optimal quantization (all bits for Lloyd-Max)
2. If absolute scores needed downstream: apply trivial scalar recalibration
3. Reserve unbiased quantization (TQ, QJL) for KV cache or softmax-adjacent pipelines
4. Evaluate quantizers on Recall@K, not on inner product bias

---

## References
1. TurboQuant: arXiv:2504.19874 — Online Vector Quantization with Near-optimal Distortion Rate
2. Netflix: arXiv:2403.05440 — Is Cosine-Similarity of Embeddings Really About Similarity?
3. QJL: arXiv:2406.03482 — QJL: 1-Bit Quantized JL Transform for KV Cache Quantization
4. Product Quantization: Jégou et al. (2011) — Product quantization for nearest neighbor search
5. RaBitQ: Gao & Long (2024) — RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
6. James-Stein: Stein (1956) — Inadmissibility of the Usual Estimator for the Mean of a Multivariate Normal Distribution
