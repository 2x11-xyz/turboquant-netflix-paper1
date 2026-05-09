"""
4-bit and 8-bit dimension sweep (parallelized).
Critical test: Does MSE still dominate when TQ only pays 25% (4-bit) or 12.5% (8-bit) for QJL?

Run: `modal run dim_sweep_4bit.py`
"""
import modal

app = modal.App("tq-4bit-sweep")
volume = modal.Volume.from_name("turboquant-netflix-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scipy")
    .add_local_file("turboquant_impl.py", "/root/turboquant_impl.py")
)

DIMS = [50, 128, 256, 512, 1024, 2048]
BITS_LIST = [4, 8]
N_ITEMS = 10_000
N_QUERIES = 1_000
N_SEEDS = 50


@app.function(image=image, timeout=1800, memory=16384)
def eval_seed(d, bits, seed):
    """Evaluate one (d, bits, seed) on Gaussian data."""
    import sys
    sys.path.insert(0, "/root")
    import torch
    import numpy as np
    from turboquant_impl import TurboQuantIP, MSEOnlyQuantizer

    # Generate data (deterministic per d)
    torch.manual_seed(42)
    np.random.seed(42)
    items_raw = torch.randn(N_ITEMS, d)
    items = items_raw / items_raw.norm(dim=1, keepdim=True)
    queries_raw = torch.randn(N_QUERIES, d)
    queries = queries_raw / queries_raw.norm(dim=1, keepdim=True)

    # True IPs
    true_ip = (queries @ items.T).numpy()
    n_cal = N_QUERIES // 2

    # Build quantizers
    tq = TurboQuantIP(dim=d, bits=bits, seed=seed)
    mse_q = MSEOnlyQuantizer(dim=d, bits=bits, seed=seed)

    # Quantize
    items_hat_tq = tq.dequantize(*tq.quantize(items))
    items_hat_mse = mse_q.dequantize(*mse_q.quantize(items))

    # Estimated IPs
    est_tq = (queries @ items_hat_tq.T).numpy()
    est_mse = (queries @ items_hat_mse.T).numpy()

    # Fit alpha on calibration half
    cal_true = true_ip[:n_cal].flatten()
    cal_mse = est_mse[:n_cal].flatten()
    alpha = float(np.dot(cal_mse, cal_true) / np.dot(cal_true, cal_true))
    est_deb = est_mse / alpha

    # Recall@10 on test half
    r10_tq, r10_mse, r10_deb = [], [], []
    r100_tq, r100_mse, r100_deb = [], [], []
    for q in range(n_cal, N_QUERIES):
        true_t10 = set(np.argsort(-true_ip[q])[:10])
        true_t100 = set(np.argsort(-true_ip[q])[:100])
        
        r10_tq.append(len(true_t10 & set(np.argsort(-est_tq[q])[:10])) / 10)
        r10_mse.append(len(true_t10 & set(np.argsort(-est_mse[q])[:10])) / 10)
        r10_deb.append(len(true_t10 & set(np.argsort(-est_deb[q])[:10])) / 10)
        r100_tq.append(len(true_t100 & set(np.argsort(-est_tq[q])[:100])) / 100)
        r100_mse.append(len(true_t100 & set(np.argsort(-est_mse[q])[:100])) / 100)
        r100_deb.append(len(true_t100 & set(np.argsort(-est_deb[q])[:100])) / 100)

    # MAE
    test_true = true_ip[n_cal:].flatten()
    mae_tq = float(np.abs(est_tq[n_cal:].flatten() - test_true).mean())
    mae_mse = float(np.abs(est_mse[n_cal:].flatten() - test_true).mean())
    mae_deb = float(np.abs(est_deb[n_cal:].flatten() - test_true).mean())

    # Spearman on first 50 test queries
    from scipy.stats import spearmanr
    sp_tq, sp_mse = [], []
    for q in range(n_cal, min(n_cal + 50, N_QUERIES)):
        r_t, _ = spearmanr(true_ip[q], est_tq[q])
        r_m, _ = spearmanr(true_ip[q], est_mse[q])
        sp_tq.append(r_t)
        sp_mse.append(r_m)

    return {
        "d": d, "bits": bits, "seed": seed,
        "recall10_tq": float(np.mean(r10_tq)),
        "recall10_mse": float(np.mean(r10_mse)),
        "recall10_debiased": float(np.mean(r10_deb)),
        "recall100_tq": float(np.mean(r100_tq)),
        "recall100_mse": float(np.mean(r100_mse)),
        "recall100_debiased": float(np.mean(r100_deb)),
        "mae_tq": mae_tq,
        "mae_mse": mae_mse,
        "mae_debiased": mae_deb,
        "spearman_tq": float(np.mean(sp_tq)),
        "spearman_mse": float(np.mean(sp_mse)),
        "alpha": alpha,
    }


@app.function(image=image, volumes={"/results": volume}, timeout=60)
def save_results(json_str: str):
    with open("/results/dim_sweep_4bit_results.json", "w") as f:
        f.write(json_str)
    volume.commit()
    print("Saved dim_sweep_4bit_results.json")


@app.local_entrypoint()
def main():
    import json
    import numpy as np

    # Fan out all (d, bits, seed) combinations
    call_args = []
    for d in DIMS:
        for bits in BITS_LIST:
            for seed in range(N_SEEDS):
                call_args.append((d, bits, seed))

    print(f"Launching {len(call_args)} parallel tasks ({len(DIMS)} dims × {len(BITS_LIST)} bit-widths × {N_SEEDS} seeds)...")
    results_list = list(eval_seed.starmap(call_args))
    print(f"All {len(results_list)} tasks complete!")

    # Aggregate
    results = {}
    for d in DIMS:
        results[str(d)] = {}
        for bits in BITS_LIST:
            seed_results = [r for r in results_list if r["d"] == d and r["bits"] == bits]

            agg = {}
            for key in ["recall10_tq", "recall10_mse", "recall10_debiased",
                       "recall100_tq", "recall100_mse", "recall100_debiased",
                       "mae_tq", "mae_mse", "mae_debiased",
                       "spearman_tq", "spearman_mse", "alpha"]:
                vals = [r[key] for r in seed_results]
                agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

            results[str(d)][str(bits)] = agg

    # Save
    save_results.remote(json.dumps(results, indent=2))

    # Print summary
    print(f"\n{'='*100}")
    print(f"SUMMARY: Recall@10 (4-bit and 8-bit)")
    print(f"{'='*100}")
    print(f"{'d':>6} | {'bits':>4} | {'TQ':>16} | {'MSE-only':>16} | {'Debiased':>16} | {'Gap':>8} | {'Alpha':>8}")
    print("-" * 100)
    for d in DIMS:
        for bits in BITS_LIST:
            r = results[str(d)][str(bits)]
            tq = r['recall10_tq']['mean']
            mse = r['recall10_mse']['mean']
            deb = r['recall10_debiased']['mean']
            gap = mse - tq
            alpha = r['alpha']['mean']
            print(f"{d:>6} | {bits:>4} | "
                  f"{tq:.4f}±{r['recall10_tq']['std']:.4f} | "
                  f"{mse:.4f}±{r['recall10_mse']['std']:.4f} | "
                  f"{deb:.4f}±{r['recall10_debiased']['std']:.4f} | "
                  f"{gap:>+.4f} | {alpha:.4f}")

    print(f"\n{'='*100}")
    print(f"SUMMARY: Score MAE")
    print(f"{'='*100}")
    print(f"{'d':>6} | {'bits':>4} | {'TQ MAE':>12} | {'MSE MAE':>12} | {'Debiased MAE':>12} | {'TQ/MSE ratio':>12}")
    print("-" * 80)
    for d in DIMS:
        for bits in BITS_LIST:
            r = results[str(d)][str(bits)]
            tq_mae = r['mae_tq']['mean']
            mse_mae = r['mae_mse']['mean']
            deb_mae = r['mae_debiased']['mean']
            ratio = tq_mae / mse_mae if mse_mae > 0 else float('inf')
            print(f"{d:>6} | {bits:>4} | {tq_mae:.6f}   | {mse_mae:.6f}   | {deb_mae:.6f}   | {ratio:.2f}x")
