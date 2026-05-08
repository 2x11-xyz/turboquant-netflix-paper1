"""
Critical test: Can MSE-only bias be trivially corrected by dividing by alpha?

If yes → TQ has no advantage (debiased MSE = unbiased + lower variance).
If no → TQ's formal unbiasedness has value when alpha is unstable/unknowable.

We test:
1. Fit global alpha on a calibration set, apply to held-out test set
2. Check if alpha is stable across: users, items, kappa values, seeds
3. Compare debiased-MSE vs TQ on Recall@10 and score calibration
"""
import modal

app = modal.App("tq-debiased-mse")
vol = modal.Volume.from_name("turboquant-netflix-results")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scipy")
    .add_local_file("turboquant_impl.py", "/root/turboquant_impl.py")
)

@app.function(image=image, volumes={"/results": vol}, timeout=900)
def test_debiased_mse():
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

    # Use ALL users and items for stronger evaluation
    # Split: calibration users (first 10K) vs test users (last 10K)
    U_all = (X @ A_eq1).float()  # (20000, 50)
    V_all = B_eq1.float()        # (1000, 50)

    cal_users = U_all[:10000]    # calibration set
    test_users = U_all[10000:]   # held-out test set

    n_seeds = 50

    for bits in [2, 3]:
        print(f"\n{'#'*60}")
        print(f"# {bits}-BIT: DEBIASED MSE ANALYSIS")
        print(f"{'#'*60}")

        for t in [0, 0.5, 1.0]:
            D = torch.diag(torch.exp(t * z))
            D_inv = torch.diag(torch.exp(-t * z))

            U_cal = cal_users @ D      # (10000, 50)
            U_test = test_users @ D    # (10000, 50)
            V = V_all @ D_inv          # (1000, 50)

            true_cal = (U_cal @ V.T).numpy()    # (10000, 1000)
            true_test = (U_test @ V.T).numpy()  # (10000, 1000)

            kappa = torch.exp(t * z).max().item() / torch.exp(t * z).min().item()

            print(f"\n{'='*60}")
            print(f"  bits={bits}, t={t}, κ={kappa:.1f}")
            print(f"{'='*60}")

            # --- Stability of alpha across seeds ---
            alphas_per_seed = []
            recall_tq_seeds = []
            recall_mse_seeds = []
            recall_debiased_seeds = []
            
            # Score calibration: mean absolute error on held-out
            mae_tq_seeds = []
            mae_mse_seeds = []
            mae_debiased_seeds = []

            for seed in range(n_seeds):
                tq = TurboQuantIP(dim=K, bits=bits, seed=seed)
                mse_q = MSEOnlyQuantizer(dim=K, bits=bits, seed=seed)

                # Quantize items ONCE (this is the deployed index)
                V_hat_tq = tq.dequantize(*tq.quantize(V))
                V_hat_mse = mse_q.dequantize(*mse_q.quantize(V))

                # --- Step 1: Fit alpha on calibration set ---
                cal_mse_scores = (U_cal @ V_hat_mse.T).numpy()
                
                # Global least-squares fit: mse_score ≈ alpha * true + beta
                true_cal_flat = true_cal.flatten()
                mse_cal_flat = cal_mse_scores.flatten()
                # Fit alpha only (force beta=0 since we know it's ~0)
                alpha = np.dot(mse_cal_flat, true_cal_flat) / np.dot(true_cal_flat, true_cal_flat)
                alphas_per_seed.append(alpha)

                # --- Step 2: Apply correction to TEST set ---
                test_mse_scores = (U_test @ V_hat_mse.T).numpy()
                test_tq_scores = (U_test @ V_hat_tq.T).numpy()
                test_debiased_scores = test_mse_scores / alpha  # the correction

                # --- Step 3: Evaluate on test set ---
                # Recall@10 (sample 1000 users for speed)
                eval_users = min(1000, test_mse_scores.shape[0])
                recall_tq = []
                recall_mse = []
                recall_debiased = []
                for i in range(eval_users):
                    true_top10 = set(np.argsort(-true_test[i])[:10])
                    tq_top10 = set(np.argsort(-test_tq_scores[i])[:10])
                    mse_top10 = set(np.argsort(-test_mse_scores[i])[:10])
                    debiased_top10 = set(np.argsort(-test_debiased_scores[i])[:10])
                    recall_tq.append(len(true_top10 & tq_top10) / 10)
                    recall_mse.append(len(true_top10 & mse_top10) / 10)
                    recall_debiased.append(len(true_top10 & debiased_top10) / 10)

                recall_tq_seeds.append(np.mean(recall_tq))
                recall_mse_seeds.append(np.mean(recall_mse))
                recall_debiased_seeds.append(np.mean(recall_debiased))

                # Score calibration: MAE on test set (sample for speed)
                sample_idx = np.random.choice(true_test.size, size=100000, replace=False)
                true_sample = true_test.flatten()[sample_idx]
                mae_tq_seeds.append(np.abs(test_tq_scores.flatten()[sample_idx] - true_sample).mean())
                mae_mse_seeds.append(np.abs(test_mse_scores.flatten()[sample_idx] - true_sample).mean())
                mae_debiased_seeds.append(np.abs(test_debiased_scores.flatten()[sample_idx] - true_sample).mean())

            # --- Results ---
            print(f"\n  Alpha stability across {n_seeds} seeds:")
            print(f"    mean={np.mean(alphas_per_seed):.6f}, std={np.std(alphas_per_seed):.6f}, "
                  f"min={np.min(alphas_per_seed):.6f}, max={np.max(alphas_per_seed):.6f}")

            print(f"\n  Recall@10 (held-out test, {n_seeds} seeds):")
            print(f"    TQ {bits}-bit:          {np.mean(recall_tq_seeds):.4f} ± {np.std(recall_tq_seeds):.4f}")
            print(f"    MSE-only {bits}-bit:    {np.mean(recall_mse_seeds):.4f} ± {np.std(recall_mse_seeds):.4f}")
            print(f"    Debiased MSE {bits}-bit:{np.mean(recall_debiased_seeds):.4f} ± {np.std(recall_debiased_seeds):.4f}")

            print(f"\n  Score MAE (held-out test, {n_seeds} seeds):")
            print(f"    TQ {bits}-bit:          {np.mean(mae_tq_seeds):.6f} ± {np.std(mae_tq_seeds):.6f}")
            print(f"    MSE-only {bits}-bit:    {np.mean(mae_mse_seeds):.6f} ± {np.std(mae_mse_seeds):.6f}")
            print(f"    Debiased MSE {bits}-bit:{np.mean(mae_debiased_seeds):.6f} ± {np.std(mae_debiased_seeds):.6f}")

            # --- Per-item alpha stability ---
            # Check if alpha varies by item (if so, global correction is imperfect)
            seed0_tq = TurboQuantIP(dim=K, bits=bits, seed=0)
            seed0_mse = MSEOnlyQuantizer(dim=K, bits=bits, seed=0)
            V_hat_mse_0 = seed0_mse.dequantize(*seed0_mse.quantize(V))
            
            # Per-item: for each item j, fit alpha_j from all cal users
            cal_mse_0 = (U_cal @ V_hat_mse_0.T).numpy()  # (10000, 1000)
            per_item_alpha = []
            for j in range(V.shape[0]):
                true_col = true_cal[:, j]
                mse_col = cal_mse_0[:, j]
                if np.dot(true_col, true_col) > 1e-10:
                    a_j = np.dot(mse_col, true_col) / np.dot(true_col, true_col)
                    per_item_alpha.append(a_j)

            per_item_alpha = np.array(per_item_alpha)
            print(f"\n  Per-item alpha (seed=0, {len(per_item_alpha)} items):")
            print(f"    mean={per_item_alpha.mean():.6f}, std={per_item_alpha.std():.6f}, "
                  f"min={per_item_alpha.min():.6f}, max={per_item_alpha.max():.6f}")

            # Per-user alpha stability
            per_user_alpha = []
            for i in range(min(1000, U_cal.shape[0])):
                true_row = true_cal[i]
                mse_row = cal_mse_0[i]
                if np.dot(true_row, true_row) > 1e-10:
                    a_i = np.dot(mse_row, true_row) / np.dot(true_row, true_row)
                    per_user_alpha.append(a_i)

            per_user_alpha = np.array(per_user_alpha)
            print(f"\n  Per-user alpha (seed=0, {len(per_user_alpha)} users):")
            print(f"    mean={per_user_alpha.mean():.6f}, std={per_user_alpha.std():.6f}, "
                  f"min={per_user_alpha.min():.6f}, max={per_user_alpha.max():.6f}")

    print("\nDone.")


if __name__ == "__main__":
    with app.run():
        test_debiased_mse.remote()
