"""
Check if MSE-only quantization bias is monotonic (ranking-preserving) or non-monotonic.
Also compute single-seed Recall@10 for TQ vs MSE-only.

Key questions:
1. Is MSE-only bias proportional to true score? (If so, rankings preserved despite bias)
2. What is single-seed Recall@10 for TQ vs MSE-only?
"""
import modal

app = modal.App("tq-monotonicity-check")
vol = modal.Volume.from_name("turboquant-netflix-results")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scipy", "matplotlib")
    .add_local_file("turboquant_impl.py", "/root/turboquant_impl.py")
)

@app.function(image=image, volumes={"/results": vol}, timeout=600)
def check_monotonicity():
    import sys
    sys.path.insert(0, "/root")
    import torch
    import numpy as np
    from scipy.stats import spearmanr

    data = torch.load("/results/experiment_data.pt", weights_only=False)
    X = data["X"]
    A_eq1 = data["A_eq1"]
    B_eq1 = data["B_eq1"]

    K = A_eq1.shape[1]
    N_USERS = X.shape[0]
    N_ITEMS = B_eq1.shape[0]

    from turboquant_impl import TurboQuantIP, MSEOnlyQuantizer

    torch.manual_seed(42)
    np.random.seed(42)

    z = torch.randn(K)

    # Sample users/items
    user_idx = torch.randperm(N_USERS)[:500]
    item_idx = torch.randperm(N_ITEMS)[:200]

    U_base = (X[user_idx] @ A_eq1).float()
    V_base = B_eq1[item_idx].float()

    n_seeds = 50

    for t in [0, 0.5, 1.0]:
        D = torch.diag(torch.exp(t * z))
        D_inv = torch.diag(torch.exp(-t * z))

        U = U_base @ D        # (500, K)
        V = V_base @ D_inv    # (200, K)

        true_scores = (U @ V.T).numpy()  # (500, 200)
        kappa = torch.exp(t * z).max().item() / torch.exp(t * z).min().item()

        print(f"\n{'='*60}")
        print(f"t={t}, κ={kappa:.1f}")
        print(f"{'='*60}")

        # Collect per-seed scores
        tq_all = np.zeros((n_seeds, 500, 200))
        mse_all = np.zeros((n_seeds, 500, 200))

        for seed in range(n_seeds):
            tq = TurboQuantIP(dim=K, bits=2, seed=seed)
            mse_q = MSEOnlyQuantizer(dim=K, bits=2, seed=seed)

            # Batch quantize + dequantize
            tq_codes = tq.quantize(V)
            V_hat_tq = tq.dequantize(*tq_codes)

            mse_codes = mse_q.quantize(V)
            V_hat_mse = mse_q.dequantize(*mse_codes)

            # Batch IP
            tq_all[seed] = (U @ V_hat_tq.T).numpy()
            mse_all[seed] = (U @ V_hat_mse.T).numpy()

        # --- MONOTONICITY ANALYSIS (MC-averaged) ---
        tq_mean = tq_all.mean(axis=0)   # (500, 200)
        mse_mean = mse_all.mean(axis=0)  # (500, 200)

        true_flat = true_scores.flatten()
        mse_flat = mse_mean.flatten()
        mse_bias_flat = mse_flat - true_flat

        # Global linear fit: mse_mean ≈ alpha * true + beta
        A_reg = np.column_stack([true_flat, np.ones_like(true_flat)])
        params, _, _, _ = np.linalg.lstsq(A_reg, mse_flat, rcond=None)
        alpha, beta = params

        # Relative bias (bias/true) for non-tiny scores
        mask = np.abs(true_flat) > 0.01
        relative_bias = mse_bias_flat[mask] / true_flat[mask]

        print(f"\n  MONOTONICITY (MC-averaged over {n_seeds} seeds):")
        print(f"    Global fit: MSE_estimate ≈ {alpha:.4f} * true + {beta:.6f}")
        print(f"    If multiplicative: alpha<1, beta≈0")
        print(f"    Relative bias (bias/true): mean={relative_bias.mean():.4f}, std={relative_bias.std():.4f}")
        print(f"    Absolute bias: mean={mse_bias_flat.mean():.6f}, std={mse_bias_flat.std():.6f}")
        print(f"    True score range: [{true_flat.min():.4f}, {true_flat.max():.4f}]")

        # Per-user Spearman correlation (MC-averaged scores vs true)
        spearman_mse = []
        spearman_tq = []
        for i in range(500):
            rho_mse, _ = spearmanr(true_scores[i], mse_mean[i])
            rho_tq, _ = spearmanr(true_scores[i], tq_mean[i])
            spearman_mse.append(rho_mse)
            spearman_tq.append(rho_tq)

        print(f"\n  Per-user Spearman (MC-averaged):")
        print(f"    TQ 2-bit:       mean={np.mean(spearman_tq):.4f}, min={np.min(spearman_tq):.4f}")
        print(f"    MSE-only 2-bit: mean={np.mean(spearman_mse):.4f}, min={np.min(spearman_mse):.4f}")

        # Pairwise inversion rate (MC-averaged)
        n_inv_mse = 0
        n_inv_tq = 0
        n_pairs_total = 0
        for i in range(500):
            # Sample 500 random item pairs per user
            j_pairs = np.random.randint(0, 200, size=500)
            k_pairs = np.random.randint(0, 200, size=500)
            valid = j_pairs != k_pairs
            j_pairs, k_pairs = j_pairs[valid], k_pairs[valid]

            true_diff = true_scores[i, j_pairs] - true_scores[i, k_pairs]
            mse_diff = mse_mean[i, j_pairs] - mse_mean[i, k_pairs]
            tq_diff = tq_mean[i, j_pairs] - tq_mean[i, k_pairs]

            n_inv_mse += np.sum(true_diff * mse_diff < 0)
            n_inv_tq += np.sum(true_diff * tq_diff < 0)
            n_pairs_total += len(j_pairs)

        print(f"\n  Pairwise inversion rate (MC-averaged):")
        print(f"    TQ 2-bit:       {n_inv_tq/n_pairs_total:.4f} ({n_inv_tq}/{n_pairs_total})")
        print(f"    MSE-only 2-bit: {n_inv_mse/n_pairs_total:.4f} ({n_inv_mse}/{n_pairs_total})")

        # --- SINGLE-SEED RECALL@10 (the real test) ---
        recall_tq_seeds = []
        recall_mse_seeds = []

        for seed in range(n_seeds):
            # Per-user Recall@10 from this single seed
            recall_tq_user = []
            recall_mse_user = []
            for i in range(500):
                true_top10 = set(np.argsort(-true_scores[i])[:10])
                tq_top10 = set(np.argsort(-tq_all[seed, i])[:10])
                mse_top10 = set(np.argsort(-mse_all[seed, i])[:10])
                recall_tq_user.append(len(true_top10 & tq_top10) / 10)
                recall_mse_user.append(len(true_top10 & mse_top10) / 10)

            recall_tq_seeds.append(np.mean(recall_tq_user))
            recall_mse_seeds.append(np.mean(recall_mse_user))

        print(f"\n  SINGLE-SEED Recall@10 (over {n_seeds} seeds):")
        print(f"    TQ 2-bit:       {np.mean(recall_tq_seeds):.4f} ± {np.std(recall_tq_seeds):.4f}")
        print(f"    MSE-only 2-bit: {np.mean(recall_mse_seeds):.4f} ± {np.std(recall_mse_seeds):.4f}")

        # Per-user Spearman for single seeds (distribution)
        spearman_tq_single = []
        spearman_mse_single = []
        for seed in range(n_seeds):
            rhos_tq = []
            rhos_mse = []
            for i in range(500):
                rho_tq, _ = spearmanr(true_scores[i], tq_all[seed, i])
                rho_mse, _ = spearmanr(true_scores[i], mse_all[seed, i])
                rhos_tq.append(rho_tq)
                rhos_mse.append(rho_mse)
            spearman_tq_single.append(np.mean(rhos_tq))
            spearman_mse_single.append(np.mean(rhos_mse))

        print(f"\n  SINGLE-SEED Spearman (over {n_seeds} seeds):")
        print(f"    TQ 2-bit:       {np.mean(spearman_tq_single):.4f} ± {np.std(spearman_tq_single):.4f}")
        print(f"    MSE-only 2-bit: {np.mean(spearman_mse_single):.4f} ± {np.std(spearman_mse_single):.4f}")

    print("\nDone.")


if __name__ == "__main__":
    with app.run():
        check_monotonicity.remote()
