# TurboQuant + Netflix Paper: Learnings

## The Original Hypothesis
Netflix (arXiv:2403.05440) proved cosine similarity is arbitrary under D-scaling for Eq.1-style matrix factorization. Their fix: use dot products (D-invariant). TurboQuant (arXiv:2504.19874) provides unbiased compressed dot product estimation. We hypothesized that TQ would be the ideal method to compress Netflix's D-invariant scores at scale.

## What We Found

### 1. TQ IS mathematically unbiased (confirmed)
- E[⟨u, TQ(v)⟩] = ⟨u, v⟩ — verified across all κ values, seeds, dimensions
- Clean-room implementation passes all theorem checks
- The reference library (`back2matching/turboquant`) has a scaling bug in QJL dequantization

### 2. MSE-only bias is nearly perfectly monotonic
- Global fit: MSE_estimate ≈ α·true + β, with α ≈ 0.89 (2-bit) or 0.97 (3-bit), β ≈ 0
- This means rankings are barely disturbed despite the bias
- Lloyd-Max centroids shrink coordinates → after random rotation → approximately multiplicative shrinkage on dot products

### 3. TQ's higher variance HURTS ranking (the critical finding)
Single-seed Recall@10 results (the deployment-realistic metric):

| κ | TQ 2-bit | MSE 2-bit | TQ 3-bit | MSE 3-bit |
|---|----------|-----------|----------|-----------|
| 1 | 0.682 | **0.768** | 0.761 | **0.847** |
| 7.5 | 0.552 | **0.680** | 0.657 | **0.785** |
| 56 | 0.249 | **0.403** | 0.364 | **0.548** |

MSE-only wins ranking at every operating point because:
- Its bias is monotonic (preserves order)
- Its variance is 3-5× lower per instance
- Rankings depend on the ONE deployed index, not the expectation over infinite indices

### 4. Trivial debiasing (÷α) eliminates MSE's calibration disadvantage
- Fit global α on calibration users → apply to held-out test
- Debiased MSE achieves LOWER MAE than TQ at every operating point
- TQ's variance penalty overwhelms its unbiasedness advantage even for absolute score accuracy

Score MAE comparison:

| Setting | TQ | MSE-only | Debiased MSE |
|---------|-----|----------|--------------|
| 2-bit, κ=1 | 0.0100 | 0.0060 | **0.0050** |
| 3-bit, κ=1 | 0.0057 | 0.0028 | **0.0027** |

### 5. Per-item α is NOT stable (a glimmer of hope for TQ)
- Per-item α std: 0.06-0.69 depending on bits and κ
- Range at κ=56, 2-bit: [-2.0, 3.5] — wildly unstable
- But global correction still dominates because the resulting variance is still less than TQ's variance
- This instability might matter for cold-start items (no calibration data)

## Why Unbiasedness Doesn't Win Here

The fundamental insight: **unbiasedness is an expectation-level property; deployment is a single-instance reality.**

TQ guarantees E[estimate] = true score. But you deploy ONE quantized index. That one instance:
- Has zero bias on average (across parallel universes)
- Has HIGH variance (this specific universe might be way off)

MSE-only:
- Has predictable, monotonic bias (all scores shrink by ~11%)
- Has LOW variance (this specific universe is close to the biased prediction)
- The bias can be corrected post-hoc with a trivial scalar

It's the James-Stein phenomenon applied to vector quantization: a biased shrinkage estimator can dominate an unbiased one in MSE when the bias is structured and predictable.

## When TQ Might Still Win (Untested)

1. **Cold-start items with no calibration data** — α is unknown for new items. TQ needs no calibration by construction. (But MSE-only's global α might be good enough.)

2. **Non-stationary embeddings** — if the model retrains and α shifts, TQ's guarantee holds without recalibration. MSE-only needs periodic α updates.

3. **Very high dimensions (d=512+)** — TQ variance scales as 1/d. At production dimensions, the variance gap may narrow enough for TQ to become competitive. (Untested here; d=50 is artificially low.)

4. **Adversarial D-scaling** — at extreme κ, per-item α becomes wildly unstable. If your MF model produces pathological D-scalings, global correction fails. But at that point, ALL methods degrade.

5. **Multi-quantizer ensembles** — averaging multiple TQ instances (different seeds) would reduce variance while preserving unbiasedness. With k=4 seeds, variance drops 4×. But storage also grows 4×.

## Technical Discoveries

### TurboQuant library bug
- `TurboQuantIP.dequantize()` uses `sqrt(pi/2)/dim` for QJL coefficient
- S matrix initialized with N(0, 1/d) entries
- Correct (per paper Definition 1): N(0,1) entries with `sqrt(pi/2)/d` coefficient
- Confirmed with z-scores −8 to −24 across dims/bits/seeds
- Clean-room implementation verified unbiased (all |z| ≤ 2.3)

### The MSE-only debiasing identity
For random rotation Π + Lloyd-Max scalar quantization:
- Each coordinate is independently shrunk toward zero
- After rotation back, the net effect on dot products is approximately multiplicative
- The multiplicative factor depends on the quantizer's distortion characteristics, not on the data
- This makes it approximately knowable a priori (from the codebook alone, without ground truth)

## Implications for Paper 2 (Incremental Item Addition)

The original Paper 2 plan was: "TQ enables real-time item addition because it's data-oblivious." But MSE-only is equally data-oblivious (same random rotation architecture). So TQ's data-oblivious design is not a unique advantage.

The remaining angle for Paper 2: new items have unknown per-item α. If per-item calibration matters (which our results suggest it doesn't much, since global α dominates), then TQ's formal guarantee might help for cold-start items. But this is a weak argument.

## Verdict

**TQ's unbiasedness is a mathematical luxury that doesn't translate to practical advantage in this setting.** At d=50, the variance cost of unbiasedness overwhelms its theoretical benefit. MSE-only + trivial debiasing dominates on both ranking and calibration.

This is a useful negative result — it saves practitioners from deploying TQ thinking "unbiased = better." It might be worth a blog post or technical report, but not a workshop paper in its current form.

## Files & Artifacts
- Repo: https://github.com/2x11-xyz/turboquant-netflix-paper1
- Clean-room TQ impl: `turboquant_impl.py`
- Experiment data: Modal volume `turboquant-netflix-results/experiment_data.pt`
- Analysis scripts: `analysis_monotonicity.py`, `analysis_3bit.py`, `analysis_debiased_mse.py`
- Figure: `figure1_scatter.pdf` (3×4 scatter showing TQ vs MSE-only)
- Outline (now outdated): `PAPER1_OUTLINE.md`
