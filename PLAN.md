# Execution Plan: "Unbiased Quantization Does Not Improve Vector Retrieval"

## Goal
Show that MSE-optimal quantization dominates TurboQuant for top-K retrieval across dimensionalities and datasets. Or, find the crossover point where TQ's 1/d variance advantage kicks in.

---

## Phase 1: Dimension Sweep (Synthetic)
**Purpose**: Find where TQ becomes competitive as d grows.

**Setup**:
- Random Gaussian embeddings (controlled, no confounders)
- d ∈ {50, 128, 256, 512, 1024, 2048}
- n_items = 10,000; n_queries = 1,000
- bits ∈ {2, 3}
- 50 seeds per condition
- Metrics: Recall@10, Recall@100, Spearman, MAE, monotonicity α

**Key question**: At what d does TQ's Recall@10 match MSE-only?

**Implementation**: Single Modal script, Gaussian data generated on the fly (no volume needed).

---

## Phase 2: Real Datasets
**Purpose**: Validate findings on TQ's own benchmarks.

### Dataset 1: GloVe-200 (d=200)
- 1.2M word vectors, standard ANN benchmark
- TQ paper uses this; we replicate their setting
- Query set: 10K random vectors from the corpus

### Dataset 2: DBPedia/OpenAI embeddings (d=1536)
- TQ paper's primary high-d benchmark
- If available via their repo or standard download
- This is the critical test: if MSE-only wins here, the paper is airtight

### Dataset 3 (optional): Sentence-Transformers (d=384 or d=768)
- all-MiniLM-L6-v2 or similar
- Represents the production retrieval sweet spot

**Metrics**: Same as Phase 1 + comparison to TQ paper's reported numbers.

---

## Phase 3: Fair Bit-Budget Accounting
**Purpose**: Address the storage overhead objection.

Account explicitly for each method:
- **MSE-only b-bit**: b·d bits + 32 bits (norm) per vector
- **TQ b-bit**: (b-1)·d bits (MSE) + d bits (QJL signs) + 32 bits (norm) + 32 bits (residual norm) = b·d + 64 bits

At d=50: TQ overhead is 64/150 = 43% more metadata. At d=1536: 64/4608 = 1.4% overhead — negligible.

Also compare: "TQ at b+1 bits vs MSE at b bits" to quantify how many extra bits TQ needs to compensate.

---

## Phase 4: Tighten the Theory
**Purpose**: Replace hand-waving with precise statements.

1. **Prove α ≈ 1 - D_mse**: Lloyd-Max distortion per coordinate under Gaussian marginals → after rotation, dot product shrinkage = 1 minus normalized MSE. Tie to TQ's own Theorem 1 bound.

2. **Formalize ranking preservation**: Under multiplicative bias α with residual noise ε, derive pairwise inversion probability as function of score gap and variance. Show MSE-only's lower variance yields fewer inversions.

3. **Drop James-Stein analogy** or make it precise. Better framing: "pointwise vs pairwise error."

---

## Phase 5: Paper Writing
**Structure** (4 pages):

1. **Intro** (0.5p): TQ claims unbiasedness helps retrieval → we show it doesn't (at least in low-to-moderate d). Frame as ablation gap in TQ paper.
2. **Background** (0.5p): MSE-only setup, TQ setup, the variance cost of unbiasedness.
3. **Why monotonic bias beats zero bias** (0.75p): α ≈ 1 - D_mse, ranking preservation, debiased MSE.
4. **Experiments** (1.5p): Dimension sweep + real datasets + bit-budget analysis.
5. **Discussion** (0.75p): When TQ wins (KV cache, softmax), limitations, recommendations.

---

## Phase 6: Submission
- Post to arXiv (~May 20-25)
- Target RecSys 2026 R&P Notes (deadline July 15)
- Backup: NeurIPS 2026 Workshops (deadline ~Sep)

---

## Execution Order

| Step | Task | Dependencies | Est. Time |
|------|------|--------------|-----------|
| 1 | Dimension sweep script (Gaussian) | None | 1 hour code + 10 min run |
| 2 | Download GloVe-200 + run experiment | None | 1 hour |
| 3 | Find/download DBPedia-1536 embeddings | Check TQ repo | 1-2 hours |
| 4 | Run real-dataset experiments | Steps 2-3 | 30 min run |
| 5 | Bit-budget analysis table | Steps 1-4 results | 30 min |
| 6 | Tighten α ≈ 1 - D_mse theory | None (pen and paper) | 1 hour |
| 7 | Update outline with all results | Steps 1-5 | 1 hour |
| 8 | Science swarm on updated outline | Step 7 | 30 min |
| 9 | Write LaTeX draft | Step 8 | 3-4 hours |
| 10 | Final science swarm on paper | Step 9 | 30 min |
| 11 | ArXiv submission | Step 10 | 30 min |

**Total estimated**: ~2-3 days of focused work.

---

## Key Risks

1. **MSE-only loses at high d** → Paper becomes "regime boundary" paper (still publishable but weaker title)
2. **GloVe/DBPedia show TQ winning** → Even more interesting finding; shows d-dependence is the real story
3. **Per-item α instability on real data** → Might restore some TQ advantage for calibration
4. **Can't access TQ's exact benchmark data** → Use GloVe (publicly available) as primary

---

## Success Criteria

**Strong paper**: MSE-only wins on Recall@10 at d ≤ 512, narrows at d ≥ 1024. Real data confirms.
**Good paper**: Clear crossover point identified with theory explanation (1/d variance vs fixed bias).
**Weak paper**: MSE-only only wins at d=50, result is trivial/known. → Probably don't publish.

---

## Files to Produce

- `dim_sweep.py` — Modal script for Gaussian dimension sweep
- `glove_experiment.py` — Modal script for GloVe-200
- `highd_experiment.py` — Modal script for d=1536 embeddings
- `PAPER1_OUTLINE_v2.md` — Updated outline with all results
- `paper.tex` — Final LaTeX
- `figures/` — All publication-ready figures
