# Paper 1: The Bias-Variance Tradeoff in Compressing D-Invariant Recommender Scores
**Working Title**: Compressing Scale-Invariant Recommender Scores: An Unbiasedness-Ranking Tradeoff  
**Target**: RecSys Workshop / NeurIPS Efficient ML Workshop (Short Paper, 4 pages max)

---

## 1. Introduction (0.5 page)
- Netflix (2403.05440) proved cosine similarity of learned embeddings is **arbitrary**: equivalent models under diagonal rescaling D produce different cosine values. Their fix: use raw dot products, which are D-invariant.
- **The production gap**: Serving full-precision dot products at billion-item scale is expensive. Compression is necessary.
- **The natural assumption**: An unbiased compressed estimator (one that preserves scores in expectation) should be superior to a biased one. TurboQuant (2504.19874) provides exactly this — a near-optimal unbiased quantizer for inner products.
- **Our finding**: This assumption is wrong for ranking. We characterize a fundamental bias-variance tradeoff when compressing D-invariant scores:
  - **MSE-only quantization**: Biased (systematic ~3-11% shrinkage), but low variance → **preserves rankings well**
  - **TurboQuant**: Unbiased (preserves scores in expectation), but higher variance → **worse rankings per deployment**
  - The bias is nearly perfectly monotonic (multiplicative), so it barely disturbs item orderings
- **Our contribution**: We document this tradeoff, show when each method wins, and provide practitioners a decision framework.

**When to use which:**

| Approach | Ranking quality | Score calibration | Memory |
|---|---|---|---|
| Full precision dot product | Best | Best | Worst |
| MSE-only b-bit | Good (low variance, monotonic bias) | Bad (systematic shrinkage) | Good |
| TQ b-bit (unbiased) | Worse (high variance per instance) | Best (unbiased in expectation) | Good |

**Implication**: Use MSE-only for top-K retrieval. Use TQ for any downstream task where absolute score values matter: CTR prediction, bid pricing, score thresholding, multi-model blending, A/B testing, explore/exploit bandits, revenue forecasting.

*Note: TurboQuant is not the first unbiased compressed estimator — QJL (Zandieh et al., 2024) provides unbiased 1-bit estimation. TurboQuant achieves near-optimal distortion at arbitrary bit-widths.*

---

## 2. Background (0.75 page)

### 2.1 Netflix's D-Scaling Result
- Model: $X \approx XAB^\top$ with two regularization schemes:
  - **Eq.1**: $\min_{A,B} \|X - XAB^\top\|_F^2 + \lambda \|AB^\top\|_F^2$ — invariant to diagonal scaling D
  - **Eq.2**: $\min_{A,B} \|X - XAB^\top\|_F^2 + \lambda(\|XA\|_F^2 + \|B\|_F^2)$ — unique solution
- For Eq.1: if $\hat{A}\hat{B}^\top$ is a solution, so is $\hat{A}D \cdot D^{-1}\hat{B}^\top$ for any invertible diagonal D
- **Cosine similarity is arbitrary** under D. **User-item dot product is invariant**: $\langle u^{(D)}, v^{(D)} \rangle = \langle u, v \rangle$
- Netflix recommends dot products — but is silent on compression

### 2.2 The Compression Question
- At billion-item scale, full-precision embeddings require ~200GB (d=50, float32, 1B items)
- 2-3 bit compression reduces this to ~20-30GB (single GPU)
- Two paradigms for dot product estimation from compressed vectors:
  - **MSE-optimal**: Minimize reconstruction error (Lloyd-Max quantization). No unbiasedness guarantee.
  - **Unbiased**: TurboQuant — sacrifices some MSE for a formal $\mathbb{E}[\hat{s}] = s$ guarantee.

### 2.3 TurboQuant's Unbiasedness Guarantee
TurboQuant quantizes items (one-sided); user queries remain full-precision:
$$\mathbb{E}[\langle u, \text{DeQuant}(Q(v)) \rangle] = \langle u, v \rangle \quad \text{(exactly unbiased, Theorem 2)}$$

Combined with Netflix's D-invariance:
$$\mathbb{E}[\langle u^{(D)}, \text{DeQuant}(Q(v^{(D)})) \rangle] = \langle u^{(D)}, v^{(D)} \rangle = \langle u, v \rangle$$

Variance bound: $\text{Var} \leq O(\|u^{(D)}\|^2 \cdot \|v^{(D)}\|^2 / d \cdot 4^{-b})$

---

## 3. The Bias-Variance Tradeoff (1 page)

### 3.1 MSE-Only Bias is Monotonic
We find empirically that MSE-only quantization produces a nearly perfectly multiplicative bias:
$$\hat{s}_{\text{MSE}} \approx \alpha \cdot s_{\text{true}} + \beta, \quad \alpha \approx 0.89 \text{ (2-bit)}, \quad \alpha \approx 0.97 \text{ (3-bit)}, \quad \beta \approx 0$$

**Why**: Lloyd-Max centroids for Gaussian-like marginals shrink coordinates toward zero. After random rotation, each coordinate's reconstruction undershoots proportionally. The rotation distributes this uniformly across all directions, making the total shrinkage approximately multiplicative on the dot product.

**Consequence for ranking**: Monotonic transformations preserve orderings. If all scores shrink by the same factor, top-K retrieval is unaffected.

### 3.2 TQ Variance Costs Ranking Quality
TurboQuant's QJL correction eliminates bias but adds variance. In a single deployed index (one random seed):
- Each item's reconstructed vector has a random perturbation from truth
- These perturbations are independent across items
- For closely-scored items, the perturbation can exceed the score gap → rank inversions

**Empirical ranking results (single-seed Recall@10, 500 users × 200 items, 50 seeds):**

| κ(D) | TQ 2-bit | MSE 2-bit | TQ 3-bit | MSE 3-bit |
|------|----------|-----------|----------|-----------|
| 1 | 0.795 | **0.871** | 0.861 | **0.921** |
| 7.5 | 0.670 | **0.801** | 0.780 | **0.883** |
| 56 | 0.330 | **0.506** | 0.468 | **0.682** |

MSE-only wins ranking at every operating point. The gap widens with κ (more variance from D-scaling).

**Single-seed Spearman correlation:**

| κ(D) | TQ 2-bit | MSE 2-bit | TQ 3-bit | MSE 3-bit |
|------|----------|-----------|----------|-----------|
| 1 | 0.781 | **0.903** | 0.883 | **0.965** |
| 7.5 | 0.623 | **0.801** | 0.770 | **0.913** |
| 56 | 0.266 | **0.462** | 0.419 | **0.662** |

### 3.3 When Unbiasedness Wins
TQ's unbiasedness is critical when absolute score values matter (not just ordering):

1. **CTR/Conversion prediction** — score = P(click). 11% shrinkage → systematically wrong predictions
2. **Ad auction bidding** — bid = P(conversion) × payout. Underbid by 11% → lost revenue
3. **Score thresholding** — "recommend if score > 0.5". Boundary items filtered incorrectly
4. **Multi-model blending** — shrinkage distorts blend weights across heterogeneous models
5. **Explore/exploit (bandits)** — miscalibrated confidence intervals → suboptimal exploration
6. **A/B testing** — systematic score distortion confounds measured treatment effects

### 3.4 The Prescriptive Link to Eq.2
Regardless of which quantizer is chosen, variance scales with $\|u\|^2 \cdot \|v\|^2$. Netflix's Eq.2 regularization ($\|XA\|_F^2 + \|B\|_F^2$) directly bounds these norms, providing variance control. This motivates:
- Use Eq.2-style weight decay if you plan to quantize (either method)
- Eq.1's unbounded D-scaling can make BOTH methods degrade (κ > 50 hurts both)

---

## 4. Experiments (0.75 page)

### 4.1 Setup
Following Netflix's synthetic data (Section 4):
- n=20,000 users, p=1,000 items, C=5 clusters, k=50
- Power-law popularity/activity, Dirichlet user preferences
- D-scalings: $D = \text{diag}(e^{tz})$, $t \in \{0, 0.5, 1\}$, giving κ ∈ {1, 7.5, 56}
- 50 random quantizer seeds per operating point
- Clean-room TurboQuant implementation (reference library has a scaling bug)

### 4.2 Figure 1: Scatter Plots (3 rows × 4 columns)
- Rows: κ ∈ {1, 7.5, 56}
- Columns: TQ 2-bit, MSE-only 2-bit, TQ 3-bit, MSE-only 3-bit
- Each dot = one (user, item) pair; y-axis = MC mean over 100 seeds; x = true score
- TQ (blue): centered on y=x. MSE-only (red): systematically below y=x
- Demonstrates unbiasedness visually, but also shows TQ's greater spread
- Note: At high κ, TQ's error is heteroscedastic and positively skewed (Var ∝ ‖u‖²·‖v‖², non-negative support compresses downward tail)

### 4.3 Table 1: Bias-Variance Summary
| Method | Bits | Bias | Variance | Recall@10 (κ=1) | Recall@10 (κ=56) |
|--------|------|------|----------|-----------------|-------------------|
| TQ | 2 | ~0 | 1.0e-3 | 0.795 | 0.330 |
| MSE-only | 2 | -0.004 | 1.9e-4 | 0.871 | 0.506 |
| TQ | 3 | ~0 | 3.2e-4 | 0.861 | 0.468 |
| MSE-only | 3 | -0.001 | 6.7e-5 | 0.921 | 0.682 |

### 4.4 Monotonicity Verification
- Linear fit: MSE_estimate ≈ α·true + β with α ∈ {0.89, 0.97}, β ≈ 0
- Pairwise inversion rate: MSE-only ~2-4% vs TQ ~4-7% (at κ=1)
- MSE-only bias is almost perfectly ranking-preserving

---

## 5. Discussion (0.5 page)

### 5.1 Practical Guidance
- **For top-K retrieval**: Use MSE-only quantization. Lower variance → better rankings. The monotonic bias doesn't affect which items appear in your top-K.
- **For score-dependent decisions**: Use TQ. CTR prediction, bid pricing, thresholding, blending, A/B testing — anywhere the absolute score value feeds into a downstream computation.
- **Hybrid architecture**: Use MSE-only for ANN candidate retrieval (top-100), then re-score top candidates with TQ (or full precision) for calibrated final scores.

### 5.2 The Role of Eq.2
- Both methods degrade with large κ (D-scaling stretches variance for all quantizers)
- Eq.2's norm regularization bounds variance regardless of quantizer choice
- Practical recommendation: if you plan to quantize item embeddings, prefer Eq.2-style training or explicit norm regularization

### 5.3 Limitations
- Theory is for linear MF. Deep two-tower models have approximate D-scaling; TQ's formal guarantee still holds but the Netflix invariance argument is weaker.
- Unbiasedness is over quantizer randomness; single deployment is one draw.
- d=50 is lower than production (128-512). TQ variance scales as 1/d, so the ranking gap narrows at production scale.
- One-sided compression only (items quantized, not users).
- Synthetic data only; no real-world retrieval benchmark.

### 5.4 Future Work
- Paper 2: TurboQuant's data-oblivious design enables incremental item addition without codebook retraining
- Production-scale evaluation at d=128-512 (where variance gap narrows)
- Hybrid retrieval pipeline evaluation (MSE-only ANN + TQ re-scoring)

---

## References
1. Netflix: arXiv:2403.05440 — Is Cosine-Similarity of Embeddings Really About Similarity?
2. TurboQuant: arXiv:2504.19874 — Online Vector Quantization with Near-optimal Distortion Rate
3. QJL: arXiv:2406.03482 — QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead
4. Product Quantization: Jégou et al. (2011) — Product quantization for nearest neighbor search
5. RaBitQ: Gao & Long (2024) — RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
