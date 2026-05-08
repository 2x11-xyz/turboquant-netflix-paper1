"""Same analysis but at 3-bit — does TQ beat MSE-only for ranking when variance is lower?"""
import modal

app = modal.App("tq-3bit-ranking")
vol = modal.Volume.from_name("turboquant-netflix-results")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scipy")
    .add_local_file("turboquant_impl.py", "/root/turboquant_impl.py")
)

@app.function(image=image, volumes={"/results": vol}, timeout=600)
def check_3bit():
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

    from turboquant_impl import TurboQuantIP, MSEOnlyQuantizer

    torch.manual_seed(42)
    np.random.seed(42)
    z = torch.randn(K)

    user_idx = torch.randperm(X.shape[0])[:500]
    item_idx = torch.randperm(B_eq1.shape[0])[:200]

    U_base = (X[user_idx] @ A_eq1).float()
    V_base = B_eq1[item_idx].float()

    n_seeds = 50

    for bits in [2, 3]:
        print(f"\n{'#'*60}")
        print(f"# {bits}-BIT COMPARISON")
        print(f"{'#'*60}")

        for t in [0, 0.5, 1.0]:
            D = torch.diag(torch.exp(t * z))
            D_inv = torch.diag(torch.exp(-t * z))
            U = U_base @ D
            V = V_base @ D_inv
            true_scores = (U @ V.T).numpy()
            kappa = torch.exp(t * z).max().item() / torch.exp(t * z).min().item()

            tq_all = np.zeros((n_seeds, 500, 200))
            mse_all = np.zeros((n_seeds, 500, 200))

            for seed in range(n_seeds):
                tq = TurboQuantIP(dim=K, bits=bits, seed=seed)
                mse_q = MSEOnlyQuantizer(dim=K, bits=bits, seed=seed)

                V_hat_tq = tq.dequantize(*tq.quantize(V))
                V_hat_mse = mse_q.dequantize(*mse_q.quantize(V))

                tq_all[seed] = (U @ V_hat_tq.T).numpy()
                mse_all[seed] = (U @ V_hat_mse.T).numpy()

            # Single-seed metrics
            recall_tq = []
            recall_mse = []
            spearman_tq = []
            spearman_mse = []

            for seed in range(n_seeds):
                r_tq = []
                r_mse = []
                s_tq = []
                s_mse = []
                for i in range(500):
                    true_top10 = set(np.argsort(-true_scores[i])[:10])
                    tq_top10 = set(np.argsort(-tq_all[seed, i])[:10])
                    mse_top10 = set(np.argsort(-mse_all[seed, i])[:10])
                    r_tq.append(len(true_top10 & tq_top10) / 10)
                    r_mse.append(len(true_top10 & mse_top10) / 10)
                    rho_tq, _ = spearmanr(true_scores[i], tq_all[seed, i])
                    rho_mse, _ = spearmanr(true_scores[i], mse_all[seed, i])
                    s_tq.append(rho_tq)
                    s_mse.append(rho_mse)
                recall_tq.append(np.mean(r_tq))
                recall_mse.append(np.mean(r_mse))
                spearman_tq.append(np.mean(s_tq))
                spearman_mse.append(np.mean(s_mse))

            # Monotonicity check
            mse_mean = mse_all.mean(axis=0)
            true_flat = true_scores.flatten()
            mse_flat = mse_mean.flatten()
            A_reg = np.column_stack([true_flat, np.ones_like(true_flat)])
            params, _, _, _ = np.linalg.lstsq(A_reg, mse_flat, rcond=None)
            alpha, beta = params

            print(f"\n  t={t}, κ={kappa:.1f}, bits={bits}")
            print(f"    MSE fit: estimate ≈ {alpha:.4f} * true + {beta:.6f}")
            print(f"    Single-seed Recall@10:  TQ={np.mean(recall_tq):.4f}±{np.std(recall_tq):.4f}  MSE={np.mean(recall_mse):.4f}±{np.std(recall_mse):.4f}")
            print(f"    Single-seed Spearman:   TQ={np.mean(spearman_tq):.4f}±{np.std(spearman_tq):.4f}  MSE={np.mean(spearman_mse):.4f}±{np.std(spearman_mse):.4f}")

    print("\nDone.")

if __name__ == "__main__":
    with app.run():
        check_3bit.remote()
