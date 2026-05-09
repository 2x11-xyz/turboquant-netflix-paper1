"""
Phase 1: Dimension Sweep on Random Gaussian Embeddings

Tests MSE-only vs TurboQuant at d ∈ {50, 128, 256, 512, 1024, 2048}
with bits ∈ {2, 3} and 50 seeds per condition.

Key question: At what d does TQ's Recall@10 match MSE-only?

Run: `modal run dim_sweep.py`
"""
import modal

app = modal.App("tq-dim-sweep")
volume = modal.Volume.from_name("turboquant-netflix-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scipy")
    .add_local_file("turboquant_impl.py", "/root/turboquant_impl.py")
)

DIMS = [50, 128, 256, 512, 1024, 2048]
BITS_LIST = [2, 3]
N_ITEMS = 10_000
N_QUERIES = 1_000
N_SEEDS = 50
TOP_K = [10, 100]


@app.function(image=image, volumes={"/results": volume}, timeout=7200, memory=32768)
def run_dim_sweep():
    import sys
    sys.path.insert(0, "/root")
    import torch
    import numpy as np
    import json
    import time
    from turboquant_impl import TurboQuantIP, MSEOnlyQuantizer

    results = {}

    for d in DIMS:
        results[d] = {}
        print(f"\n{'#'*70}")
        print(f"# DIMENSION d={d}")
        print(f"{'#'*70}")

        # Generate random Gaussian embeddings (controlled)
        torch.manual_seed(42)
        np.random.seed(42)

        # Items: random unit vectors (standard for ANN benchmarks)
        items_raw = torch.randn(N_ITEMS, d)
        items = items_raw / items_raw.norm(dim=1, keepdim=True)

        # Queries: random unit vectors
        queries_raw = torch.randn(N_QUERIES, d)
        queries = queries_raw / queries_raw.norm(dim=1, keepdim=True)

        # True inner products
        true_ip = (queries @ items.T).numpy()  # (N_QUERIES, N_ITEMS)

        # True top-K per query
        true_top10 = [set(np.argsort(-true_ip[q])[:10]) for q in range(N_QUERIES)]
        true_top100 = [set(np.argsort(-true_ip[q])[:100]) for q in range(N_QUERIES)]

        for bits in BITS_LIST:
            t_start = time.time()
            print(f"\n  --- {bits}-bit ---")

            recall10_tq = []
            recall100_tq = []
            recall10_mse = []
            recall100_mse = []
            recall10_debiased = []
            recall100_debiased = []
            mae_tq = []
            mae_mse = []
            mae_debiased = []
            spearman_tq = []
            spearman_mse = []
            alpha_values = []
            inversion_rate_tq = []
            inversion_rate_mse = []

            for seed in range(N_SEEDS):
                # Build quantizers
                tq = TurboQuantIP(dim=d, bits=bits, seed=seed)
                mse_q = MSEOnlyQuantizer(dim=d, bits=bits, seed=seed)

                # Quantize items (one-sided, asymmetric)
                items_hat_tq = tq.dequantize(*tq.quantize(items))
                items_hat_mse = mse_q.dequantize(*mse_q.quantize(items))

                # Compute estimated IPs
                est_tq = (queries @ items_hat_tq.T).numpy()
                est_mse = (queries @ items_hat_mse.T).numpy()

                # --- Fit alpha on first half of queries (calibration) ---
                cal_true = true_ip[:500].flatten()
                cal_mse = est_mse[:500].flatten()
                alpha = np.dot(cal_mse, cal_true) / np.dot(cal_true, cal_true)
                alpha_values.append(alpha)

                # Apply correction to second half (test)
                est_debiased = est_mse / alpha

                # --- Recall@K (on test queries: 500-1000) ---
                r10_tq = []
                r100_tq = []
                r10_mse = []
                r100_mse = []
                r10_deb = []
                r100_deb = []
                for q in range(500, N_QUERIES):
                    tq_top10 = set(np.argsort(-est_tq[q])[:10])
                    mse_top10 = set(np.argsort(-est_mse[q])[:10])
                    deb_top10 = set(np.argsort(-est_debiased[q])[:10])
                    tq_top100 = set(np.argsort(-est_tq[q])[:100])
                    mse_top100 = set(np.argsort(-est_mse[q])[:100])
                    deb_top100 = set(np.argsort(-est_debiased[q])[:100])

                    r10_tq.append(len(true_top10[q] & tq_top10) / 10)
                    r100_tq.append(len(true_top100[q] & tq_top100) / 100)
                    r10_mse.append(len(true_top10[q] & mse_top10) / 10)
                    r100_mse.append(len(true_top100[q] & mse_top100) / 100)
                    r10_deb.append(len(true_top10[q] & deb_top10) / 10)
                    r100_deb.append(len(true_top100[q] & deb_top100) / 100)

                recall10_tq.append(np.mean(r10_tq))
                recall100_tq.append(np.mean(r100_tq))
                recall10_mse.append(np.mean(r10_mse))
                recall100_mse.append(np.mean(r100_mse))
                recall10_debiased.append(np.mean(r10_deb))
                recall100_debiased.append(np.mean(r100_deb))

                # --- MAE (on test queries) ---
                test_true = true_ip[500:].flatten()
                test_tq = est_tq[500:].flatten()
                test_mse_flat = est_mse[500:].flatten()
                test_deb = est_debiased[500:].flatten()
                mae_tq.append(np.abs(test_tq - test_true).mean())
                mae_mse.append(np.abs(test_mse_flat - test_true).mean())
                mae_debiased.append(np.abs(test_deb - test_true).mean())

                # --- Spearman (first 100 test queries) ---
                from scipy.stats import spearmanr
                sp_tq = []
                sp_mse = []
                for q in range(500, 600):
                    r_tq, _ = spearmanr(true_ip[q], est_tq[q])
                    r_mse, _ = spearmanr(true_ip[q], est_mse[q])
                    sp_tq.append(r_tq)
                    sp_mse.append(r_mse)
                spearman_tq.append(np.mean(sp_tq))
                spearman_mse.append(np.mean(sp_mse))

                # --- Pairwise inversion rate (sample 100 pairs from 10 test queries) ---
                inv_tq = 0
                inv_mse = 0
                n_pairs = 0
                for q in range(500, 510):
                    idxs = np.random.choice(N_ITEMS, 200, replace=False)
                    for i in range(len(idxs)):
                        for j in range(i+1, min(i+10, len(idxs))):
                            ii, jj = idxs[i], idxs[j]
                            true_order = true_ip[q, ii] > true_ip[q, jj]
                            tq_order = est_tq[q, ii] > est_tq[q, jj]
                            mse_order = est_mse[q, ii] > est_mse[q, jj]
                            if true_order != tq_order:
                                inv_tq += 1
                            if true_order != mse_order:
                                inv_mse += 1
                            n_pairs += 1
                inversion_rate_tq.append(inv_tq / max(n_pairs, 1))
                inversion_rate_mse.append(inv_mse / max(n_pairs, 1))

            elapsed = time.time() - t_start
            
            results[d][bits] = {
                "recall10_tq": {"mean": float(np.mean(recall10_tq)), "std": float(np.std(recall10_tq))},
                "recall10_mse": {"mean": float(np.mean(recall10_mse)), "std": float(np.std(recall10_mse))},
                "recall10_debiased": {"mean": float(np.mean(recall10_debiased)), "std": float(np.std(recall10_debiased))},
                "recall100_tq": {"mean": float(np.mean(recall100_tq)), "std": float(np.std(recall100_tq))},
                "recall100_mse": {"mean": float(np.mean(recall100_mse)), "std": float(np.std(recall100_mse))},
                "recall100_debiased": {"mean": float(np.mean(recall100_debiased)), "std": float(np.std(recall100_debiased))},
                "mae_tq": {"mean": float(np.mean(mae_tq)), "std": float(np.std(mae_tq))},
                "mae_mse": {"mean": float(np.mean(mae_mse)), "std": float(np.std(mae_mse))},
                "mae_debiased": {"mean": float(np.mean(mae_debiased)), "std": float(np.std(mae_debiased))},
                "spearman_tq": {"mean": float(np.mean(spearman_tq)), "std": float(np.std(spearman_tq))},
                "spearman_mse": {"mean": float(np.mean(spearman_mse)), "std": float(np.std(spearman_mse))},
                "alpha": {"mean": float(np.mean(alpha_values)), "std": float(np.std(alpha_values))},
                "inversion_rate_tq": {"mean": float(np.mean(inversion_rate_tq)), "std": float(np.std(inversion_rate_tq))},
                "inversion_rate_mse": {"mean": float(np.mean(inversion_rate_mse)), "std": float(np.std(inversion_rate_mse))},
            }

            r = results[d][bits]
            print(f"    Recall@10:  TQ={r['recall10_tq']['mean']:.4f}±{r['recall10_tq']['std']:.4f}  "
                  f"MSE={r['recall10_mse']['mean']:.4f}±{r['recall10_mse']['std']:.4f}  "
                  f"Debiased={r['recall10_debiased']['mean']:.4f}±{r['recall10_debiased']['std']:.4f}")
            print(f"    Recall@100: TQ={r['recall100_tq']['mean']:.4f}±{r['recall100_tq']['std']:.4f}  "
                  f"MSE={r['recall100_mse']['mean']:.4f}±{r['recall100_mse']['std']:.4f}  "
                  f"Debiased={r['recall100_debiased']['mean']:.4f}±{r['recall100_debiased']['std']:.4f}")
            print(f"    MAE:        TQ={r['mae_tq']['mean']:.6f}  MSE={r['mae_mse']['mean']:.6f}  "
                  f"Debiased={r['mae_debiased']['mean']:.6f}")
            print(f"    Spearman:   TQ={r['spearman_tq']['mean']:.4f}  MSE={r['spearman_mse']['mean']:.4f}")
            print(f"    Alpha:      {r['alpha']['mean']:.6f} ± {r['alpha']['std']:.6f}")
            print(f"    Inversions: TQ={r['inversion_rate_tq']['mean']:.4f}  MSE={r['inversion_rate_mse']['mean']:.4f}")
            print(f"    Time: {elapsed:.1f}s")

    # Save results
    # Convert keys to strings for JSON serialization
    json_results = {}
    for d_key, d_val in results.items():
        json_results[str(d_key)] = {}
        for b_key, b_val in d_val.items():
            json_results[str(d_key)][str(b_key)] = b_val

    with open("/results/dim_sweep_results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    volume.commit()
    print(f"\n\nResults saved to /results/dim_sweep_results.json")

    # Print summary table
    print(f"\n{'='*90}")
    print(f"SUMMARY TABLE: Recall@10 (mean ± std over {N_SEEDS} seeds)")
    print(f"{'='*90}")
    print(f"{'d':>6} | {'bits':>4} | {'TQ':>14} | {'MSE-only':>14} | {'Debiased MSE':>14} | {'MSE - TQ':>10}")
    print(f"{'-'*6}-+-{'-'*4}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}-+-{'-'*10}")
    for d in DIMS:
        for bits in BITS_LIST:
            r = results[d][bits]
            tq_r = r['recall10_tq']['mean']
            mse_r = r['recall10_mse']['mean']
            deb_r = r['recall10_debiased']['mean']
            gap = mse_r - tq_r
            print(f"{d:>6} | {bits:>4} | "
                  f"{tq_r:.4f}±{r['recall10_tq']['std']:.4f} | "
                  f"{mse_r:.4f}±{r['recall10_mse']['std']:.4f} | "
                  f"{deb_r:.4f}±{r['recall10_debiased']['std']:.4f} | "
                  f"{gap:>+.4f}")

    print(f"\n{'='*90}")
    print(f"SUMMARY TABLE: Score MAE (mean over {N_SEEDS} seeds)")
    print(f"{'='*90}")
    print(f"{'d':>6} | {'bits':>4} | {'TQ':>12} | {'MSE-only':>12} | {'Debiased':>12} | {'Alpha':>10}")
    print(f"{'-'*6}-+-{'-'*4}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")
    for d in DIMS:
        for bits in BITS_LIST:
            r = results[d][bits]
            print(f"{d:>6} | {bits:>4} | "
                  f"{r['mae_tq']['mean']:.6f}   | "
                  f"{r['mae_mse']['mean']:.6f}   | "
                  f"{r['mae_debiased']['mean']:.6f}   | "
                  f"{r['alpha']['mean']:.4f}")

    return results


if __name__ == "__main__":
    with app.run():
        run_dim_sweep.remote()
