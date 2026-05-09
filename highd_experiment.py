import modal
import numpy as np

app = modal.App("tq-highd-experiment")
volume = modal.Volume.from_name("turboquant-netflix-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scipy")
    .add_local_file("turboquant_impl.py", "/root/turboquant_impl.py")
)

BITS_LIST = [2, 3]
N_SEEDS = 50
N_CLUSTERS = 20


def generate_data(d, n_items, n_queries, master_seed):
    import torch
    torch.manual_seed(master_seed)
    centers = torch.randn(N_CLUSTERS, d)
    centers = centers / centers.norm(dim=1, keepdim=True)

    def make_vectors(n):
        assignments = torch.arange(n) % N_CLUSTERS
        vecs = centers[assignments] + torch.randn(n, d) * 0.3
        return vecs / vecs.norm(dim=1, keepdim=True)

    return make_vectors(n_items), make_vectors(n_queries)


def compute_recall(true_ip, est_ip, k):
    top_true = np.argsort(-true_ip, axis=1)[:, :k]
    top_est = np.argsort(-est_ip, axis=1)[:, :k]
    return float(np.mean([len(set(t) & set(e)) / k for t, e in zip(top_true, top_est)]))


@app.function(image=image, timeout=1800, memory=16384)
def eval_single_seed(d, n_items, n_queries, master_seed, bits, seed):
    import sys
    sys.path.insert(0, "/root")
    import torch
    import numpy as np
    from scipy.stats import spearmanr
    from turboquant_impl import TurboQuantIP, MSEOnlyQuantizer

    items, queries = generate_data(d, n_items, n_queries, master_seed)
    n_cal = queries.shape[0] // 2

    tq = TurboQuantIP(dim=d, bits=bits, seed=seed)
    mse_q = MSEOnlyQuantizer(dim=d, bits=bits, seed=seed)

    items_hat_tq = tq.dequantize(*tq.quantize(items))
    items_hat_mse = mse_q.dequantize(*mse_q.quantize(items))

    # Fit alpha on calibration queries
    cal_idx = np.random.RandomState(seed).choice(n_items, min(10000, n_items), replace=False)
    cal_true = (queries[:n_cal] @ items[cal_idx].T).numpy().flatten()
    cal_mse = (queries[:n_cal] @ items_hat_mse[cal_idx].T).numpy().flatten()
    alpha = float(np.dot(cal_mse, cal_true) / np.dot(cal_true, cal_true))

    # Test set
    test_queries = queries[n_cal:]
    n_test = test_queries.shape[0]
    true_ip = (test_queries @ items.T).numpy()
    est_tq = (test_queries @ items_hat_tq.T).numpy()
    est_mse = (test_queries @ items_hat_mse.T).numpy()
    est_deb = est_mse / alpha

    # MAE on sample
    sample_size = min(100000, true_ip.size)
    idx = np.random.RandomState(seed + 1000).choice(true_ip.size, sample_size, replace=False)
    flat_true = true_ip.flatten()[idx]

    # Spearman on first 50 test queries
    sp_tq = [spearmanr(true_ip[q], est_tq[q])[0] for q in range(min(50, n_test))]
    sp_mse = [spearmanr(true_ip[q], est_mse[q])[0] for q in range(min(50, n_test))]

    return {
        "seed": seed, "bits": bits, "d": d,
        "recall10_tq": compute_recall(true_ip, est_tq, 10),
        "recall10_mse": compute_recall(true_ip, est_mse, 10),
        "recall10_debiased": compute_recall(true_ip, est_deb, 10),
        "recall100_tq": compute_recall(true_ip, est_tq, 100),
        "recall100_mse": compute_recall(true_ip, est_mse, 100),
        "recall100_debiased": compute_recall(true_ip, est_deb, 100),
        "mae_tq": float(np.abs(est_tq.flatten()[idx] - flat_true).mean()),
        "mae_mse": float(np.abs(est_mse.flatten()[idx] - flat_true).mean()),
        "mae_debiased": float(np.abs(est_deb.flatten()[idx] - flat_true).mean()),
        "spearman_tq": float(np.mean(sp_tq)),
        "spearman_mse": float(np.mean(sp_mse)),
        "alpha": alpha,
    }


@app.function(image=image, volumes={"/results": volume}, timeout=60)
def save_results(json_str: str):
    with open("/results/highd_results.json", "w") as f:
        f.write(json_str)
    volume.commit()
    print("Saved highd_results.json to volume.")


@app.local_entrypoint()
def main():
    import json

    configs = [
        {"d": 384, "n_items": 50000, "n_queries": 5000, "master_seed": 7, "name": "synthetic_384"},
        {"d": 768, "n_items": 50000, "n_queries": 5000, "master_seed": 123, "name": "synthetic_768"},
        {"d": 1536, "n_items": 50000, "n_queries": 5000, "master_seed": 42, "name": "synthetic_1536"},
    ]

    METRICS = ["recall10", "recall100", "mae", "spearman"]
    METHODS = {"recall10": ["tq", "mse", "debiased"], "recall100": ["tq", "mse", "debiased"],
               "mae": ["tq", "mse", "debiased"], "spearman": ["tq", "mse"]}

    all_results = {}
    for cfg in configs:
        d, name = cfg["d"], cfg["name"]
        print(f"\n{'='*70}\n  {name}: d={d}, items={cfg['n_items']}, queries={cfg['n_queries']}\n{'='*70}")

        call_args = [(d, cfg["n_items"], cfg["n_queries"], cfg["master_seed"], bits, seed)
                     for bits in BITS_LIST for seed in range(N_SEEDS)]
        print(f"  Launching {len(call_args)} parallel tasks...")
        results_list = list(eval_single_seed.starmap(call_args))

        setting_results = {}
        for bits in BITS_LIST:
            seed_results = [r for r in results_list if r["bits"] == bits]
            sr = {}
            for metric in METRICS:
                for method in METHODS[metric]:
                    key = f"{metric}_{method}" if method != "debiased" else f"{metric}_debiased"
                    vals = [r[key] for r in seed_results]
                    sr[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            alphas = [r["alpha"] for r in seed_results]
            sr["alpha"] = {"mean": float(np.mean(alphas)), "std": float(np.std(alphas))}
            setting_results[str(bits)] = sr

            r = sr
            print(f"\n  {bits}-bit:")
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

        all_results[name] = setting_results

    with open("/tmp/highd_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    save_results.remote(json.dumps(all_results, indent=2))

    print(f"\n\n{'='*90}\nGRAND SUMMARY: Recall@10\n{'='*90}")
    print(f"{'Setting':>20} | {'Bits':>4} | {'TQ':>14} | {'MSE-only':>14} | {'Debiased':>14} | {'Gap':>8}")
    print("-" * 90)
    for name, sr in all_results.items():
        for bits_str, r in sr.items():
            tq, mse, deb = r['recall10_tq']['mean'], r['recall10_mse']['mean'], r['recall10_debiased']['mean']
            print(f"{name:>20} | {bits_str:>4} | "
                  f"{tq:.4f}±{r['recall10_tq']['std']:.4f} | "
                  f"{mse:.4f}±{r['recall10_mse']['std']:.4f} | "
                  f"{deb:.4f}±{r['recall10_debiased']['std']:.4f} | "
                  f"{mse-tq:>+.4f}")