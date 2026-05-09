"""
Phase 2: GloVe-200 Real Dataset Experiment

Downloads GloVe 6B 200d vectors and runs MSE-only vs TurboQuant comparison.
This replicates the type of ANN benchmark used in the TurboQuant paper.

d=200, 400K items (from GloVe vocabulary), 10K queries.
bits ∈ {2, 3}, 50 seeds per condition.

Run: `modal run glove_experiment.py`
"""
import modal

app = modal.App("tq-glove-experiment")
volume = modal.Volume.from_name("turboquant-netflix-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scipy", "requests")
    .add_local_file("turboquant_impl.py", "/root/turboquant_impl.py")
)

BITS_LIST = [2, 3]
N_SEEDS = 20
N_QUERIES = 2_000
N_ITEMS = 50_000  # 50K items — representative and tractable


@app.function(image=image, volumes={"/results": volume}, timeout=14400, memory=32768)
def run_glove_experiment():
    import sys
    sys.path.insert(0, "/root")
    import torch
    import numpy as np
    import json
    import time
    import os
    import zipfile
    import requests
    from scipy.stats import spearmanr
    from turboquant_impl import TurboQuantIP, MSEOnlyQuantizer

    # --- Download GloVe if not cached ---
    glove_path = "/results/glove.6B.200d.txt"
    if not os.path.exists(glove_path):
        print("Downloading GloVe 6B (zip)...")
        url = "https://nlp.stanford.edu/data/glove.6B.zip"
        zip_path = "/tmp/glove.6B.zip"
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192*16):
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded % (100*1024*1024) == 0:
                    print(f"  Downloaded {downloaded/(1024*1024):.0f} MB / {total_size/(1024*1024):.0f} MB")
        
        print("Extracting glove.6B.200d.txt...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extract("glove.6B.200d.txt", "/results/")
        os.remove(zip_path)
        volume.commit()
        print("GloVe downloaded and cached.")
    else:
        print("GloVe already cached.")

    # --- Load GloVe vectors ---
    print("Loading GloVe vectors...")
    # GloVe 6B 200d has ~400K vectors. Load all of them.
    vectors = []
    max_load = N_ITEMS + N_QUERIES  # 390K
    with open(glove_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_load:
                break
            parts = line.strip().split()
            if len(parts) < 201:  # word + 200 dims
                continue
            vec = [float(x) for x in parts[-200:]]  # last 200 values (handles multi-word tokens)
            vectors.append(vec)
    
    vectors = torch.tensor(vectors, dtype=torch.float32)
    print(f"  Loaded {vectors.shape[0]} vectors of dim {vectors.shape[1]}")
    d = vectors.shape[1]  # should be 200

    # Split: first N_ITEMS as items, next N_QUERIES as queries
    actual_n_items = min(N_ITEMS, vectors.shape[0] - N_QUERIES)
    items = vectors[:actual_n_items]
    queries = vectors[actual_n_items:actual_n_items + N_QUERIES]
    print(f"  Split: {items.shape[0]} items, {queries.shape[0]} queries")

    # Normalize to unit vectors (standard for MIPS benchmarks)
    items = items / items.norm(dim=1, keepdim=True).clamp(min=1e-8)
    queries = queries / queries.norm(dim=1, keepdim=True).clamp(min=1e-8)

    print(f"  Items: {items.shape}, Queries: {queries.shape}")
    print(f"  Item norms: mean={items.norm(dim=1).mean():.4f}")
    print(f"  Query norms: mean={queries.norm(dim=1).mean():.4f}")

    # --- Also test with raw (un-normalized) vectors ---
    items_raw = vectors[:actual_n_items]
    queries_raw = vectors[actual_n_items:actual_n_items + N_QUERIES]

    print("\n  Raw vector stats:")
    print(f"    Item norms: mean={items_raw.norm(dim=1).mean():.4f}, "
          f"std={items_raw.norm(dim=1).std():.4f}, "
          f"min={items_raw.norm(dim=1).min():.4f}, "
          f"max={items_raw.norm(dim=1).max():.4f}")

    for setting_name, items_set, queries_set in [
        ("normalized", items, queries),
        ("raw", items_raw, queries_raw),
    ]:
        print(f"\n{'#'*70}")
        print(f"# GLOVE d={d}, setting={setting_name}")
        print(f"# Items: {items_set.shape[0]}, Queries: {queries_set.shape[0]}")
        print(f"{'#'*70}")

        # Compute true top-K (in batches to avoid OOM)
        print("  Computing true inner products (batched)...")
        n_q = queries_set.shape[0]
        n_i = items_set.shape[0]
        
        # True top-10 and top-100 per query
        true_top10 = []
        true_top100 = []
        batch_size = 500
        true_ip_test = None  # We'll store test portion for MAE
        
        for q_start in range(0, n_q, batch_size):
            q_end = min(q_start + batch_size, n_q)
            q_batch = queries_set[q_start:q_end]
            ip_batch = (q_batch @ items_set.T).numpy()  # (batch, N_ITEMS)
            
            for i in range(ip_batch.shape[0]):
                sorted_idx = np.argsort(-ip_batch[i])
                true_top10.append(set(sorted_idx[:10]))
                true_top100.append(set(sorted_idx[:100]))

        results = {}

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
            alpha_values = []

            for seed in range(N_SEEDS):
                if seed % 10 == 0:
                    print(f"    seed {seed}/{N_SEEDS}...")

                tq = TurboQuantIP(dim=d, bits=bits, seed=seed)
                mse_q = MSEOnlyQuantizer(dim=d, bits=bits, seed=seed)

                # Quantize items
                items_hat_tq = tq.dequantize(*tq.quantize(items_set))
                items_hat_mse = mse_q.dequantize(*mse_q.quantize(items_set))

                # --- Fit alpha on calibration queries (first half) ---
                n_cal = n_q // 2
                cal_item_idx = np.random.RandomState(seed).choice(n_i, min(10000, n_i), replace=False)
                cal_true = (queries_set[:n_cal] @ items_set[cal_item_idx].T).numpy().flatten()
                cal_mse = (queries_set[:n_cal] @ items_hat_mse[cal_item_idx].T).numpy().flatten()
                alpha = np.dot(cal_mse, cal_true) / np.dot(cal_true, cal_true)
                alpha_values.append(alpha)

                # --- Evaluate on test queries (second half) ---
                test_queries = queries_set[n_cal:]
                n_test = test_queries.shape[0]

                # Compute estimated IPs in batches
                r10_tq = []
                r100_tq = []
                r10_mse = []
                r100_mse = []
                r10_deb = []
                r100_deb = []
                mae_tq_batch = []
                mae_mse_batch = []
                mae_deb_batch = []

                for q_start in range(0, n_test, batch_size):
                    q_end = min(q_start + batch_size, n_test)
                    q_batch = test_queries[q_start:q_end]
                    
                    est_tq_batch = (q_batch @ items_hat_tq.T).numpy()
                    est_mse_batch = (q_batch @ items_hat_mse.T).numpy()
                    est_deb_batch = est_mse_batch / alpha

                    # True IPs for this batch
                    true_batch = (q_batch @ items_set.T).numpy()

                    for i in range(est_tq_batch.shape[0]):
                        q_global = n_cal + q_start + i  # global query index
                        
                        tq_t10 = set(np.argsort(-est_tq_batch[i])[:10])
                        mse_t10 = set(np.argsort(-est_mse_batch[i])[:10])
                        deb_t10 = set(np.argsort(-est_deb_batch[i])[:10])
                        tq_t100 = set(np.argsort(-est_tq_batch[i])[:100])
                        mse_t100 = set(np.argsort(-est_mse_batch[i])[:100])
                        deb_t100 = set(np.argsort(-est_deb_batch[i])[:100])

                        r10_tq.append(len(true_top10[q_global] & tq_t10) / 10)
                        r100_tq.append(len(true_top100[q_global] & tq_t100) / 100)
                        r10_mse.append(len(true_top10[q_global] & mse_t10) / 10)
                        r100_mse.append(len(true_top100[q_global] & mse_t100) / 100)
                        r10_deb.append(len(true_top10[q_global] & deb_t10) / 10)
                        r100_deb.append(len(true_top100[q_global] & deb_t100) / 100)

                    # MAE on a sample of IPs
                    sample_size = min(5000, est_tq_batch.size)
                    idx = np.random.choice(est_tq_batch.size, sample_size, replace=False)
                    mae_tq_batch.append(np.abs(est_tq_batch.flatten()[idx] - true_batch.flatten()[idx]).mean())
                    mae_mse_batch.append(np.abs(est_mse_batch.flatten()[idx] - true_batch.flatten()[idx]).mean())
                    mae_deb_batch.append(np.abs(est_deb_batch.flatten()[idx] - true_batch.flatten()[idx]).mean())

                recall10_tq.append(np.mean(r10_tq))
                recall100_tq.append(np.mean(r100_tq))
                recall10_mse.append(np.mean(r10_mse))
                recall100_mse.append(np.mean(r100_mse))
                recall10_debiased.append(np.mean(r10_deb))
                recall100_debiased.append(np.mean(r100_deb))
                mae_tq.append(np.mean(mae_tq_batch))
                mae_mse.append(np.mean(mae_mse_batch))
                mae_debiased.append(np.mean(mae_deb_batch))

            elapsed = time.time() - t_start

            results[bits] = {
                "recall10_tq": {"mean": float(np.mean(recall10_tq)), "std": float(np.std(recall10_tq))},
                "recall10_mse": {"mean": float(np.mean(recall10_mse)), "std": float(np.std(recall10_mse))},
                "recall10_debiased": {"mean": float(np.mean(recall10_debiased)), "std": float(np.std(recall10_debiased))},
                "recall100_tq": {"mean": float(np.mean(recall100_tq)), "std": float(np.std(recall100_tq))},
                "recall100_mse": {"mean": float(np.mean(recall100_mse)), "std": float(np.std(recall100_mse))},
                "recall100_debiased": {"mean": float(np.mean(recall100_debiased)), "std": float(np.std(recall100_debiased))},
                "mae_tq": {"mean": float(np.mean(mae_tq)), "std": float(np.std(mae_tq))},
                "mae_mse": {"mean": float(np.mean(mae_mse)), "std": float(np.std(mae_mse))},
                "mae_debiased": {"mean": float(np.mean(mae_debiased)), "std": float(np.std(mae_debiased))},
                "alpha": {"mean": float(np.mean(alpha_values)), "std": float(np.std(alpha_values))},
            }

            r = results[bits]
            print(f"    Recall@10:  TQ={r['recall10_tq']['mean']:.4f}±{r['recall10_tq']['std']:.4f}  "
                  f"MSE={r['recall10_mse']['mean']:.4f}±{r['recall10_mse']['std']:.4f}  "
                  f"Debiased={r['recall10_debiased']['mean']:.4f}±{r['recall10_debiased']['std']:.4f}")
            print(f"    Recall@100: TQ={r['recall100_tq']['mean']:.4f}±{r['recall100_tq']['std']:.4f}  "
                  f"MSE={r['recall100_mse']['mean']:.4f}±{r['recall100_mse']['std']:.4f}  "
                  f"Debiased={r['recall100_debiased']['mean']:.4f}±{r['recall100_debiased']['std']:.4f}")
            print(f"    MAE:        TQ={r['mae_tq']['mean']:.6f}  MSE={r['mae_mse']['mean']:.6f}  "
                  f"Debiased={r['mae_debiased']['mean']:.6f}")
            print(f"    Alpha:      {r['alpha']['mean']:.6f} ± {r['alpha']['std']:.6f}")
            print(f"    Time: {elapsed:.1f}s")

        # Save results for this setting
        output_key = f"glove_{setting_name}"
        output_path = f"/results/glove_{setting_name}_results.json"
        with open(output_path, "w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)
        volume.commit()
        print(f"\n  Saved {output_path}")

    return "Done"


if __name__ == "__main__":
    with app.run():
        run_glove_experiment.remote()
