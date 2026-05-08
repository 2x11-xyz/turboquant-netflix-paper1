"""
TurboQuant × Netflix experiment.
Prereq: run `modal run movie_lens_download.py` first.
Run: `modal run experiment.py`
"""
import modal

# --- Modal setup (only `modal` imported at module level to avoid local crash) ---
app = modal.App("turboquant-netflix-experiments")
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "numpy<2",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "turboquant",       # base package exports TurboQuantProd (not turboquant-pro)
        "requests",
    )
)
volume = modal.Volume.from_name("turboquant-netflix-results", create_if_missing=True)

# --- Configuration ---
K = 50          # Latent dimensions
LAMBDA = 0.01   # Regularization strength
BITS = [2, 3]   # TurboQuant bit-widths (min 2 per TurboQuantProd assertion)
T_VALS = [0, 0.5, 1, 2, 5]   # D-scaling skew levels
M_SEEDS = 10    # Different rotation seeds for TurboQuant variance estimation
N_PAIRS = 10    # User-item pairs sampled per (reg, t) condition


# ---------------------------------------------------------------------------
# Helper functions (plain Python; run inside the Modal container)
# ---------------------------------------------------------------------------

def train_mf(X, reg_scheme="eq1"):
    """Train linear MF with Netflix regularization. Returns detached (A, B)."""
    import torch

    # Fixed seed for reproducible training
    torch.manual_seed(0)

    n, p = X.shape
    A = torch.nn.Parameter(torch.randn(p, K) * 0.01)
    B = torch.nn.Parameter(torch.randn(p, K) * 0.01)
    optimizer = torch.optim.Adam([A, B], lr=0.001)

    for epoch in range(100):
        optimizer.zero_grad()
        recon = X @ A @ B.T
        loss = ((X - recon) ** 2).sum()
        if reg_scheme == "eq1":
            loss += LAMBDA * ((A @ B.T) ** 2).sum()   # ||AB^T||_F^2
        elif reg_scheme == "eq2":
            loss += LAMBDA * ((X @ A) ** 2).sum() + LAMBDA * (B ** 2).sum()
        else:
            raise ValueError(f"Unknown reg_scheme: {reg_scheme}")
        loss.backward()
        optimizer.step()

    return A.detach(), B.detach()


def generate_D(t, z):
    """Return (D, diags) for skew level t, using fixed z vector."""
    import torch

    diags = torch.exp(t * z)
    return torch.diag(diags), diags


def generate_heatmaps(results):
    """Save heatmaps: cosine sim, raw dot product, TQ variance (log scale)."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import seaborn as sns

    # results order: [eq1_t0, eq1_t0.5, ..., eq2_t0, eq2_t0.5, ...]
    # reshape (2, 5) then .T → (5, 2) so rows=t, cols=reg_scheme
    cos_data    = np.array([r["cos_sim_mean"] for r in results]).reshape(2, len(T_VALS)).T
    raw_data    = np.array([r["raw_dot_mean"] for r in results]).reshape(2, len(T_VALS)).T
    kappa_data  = np.array([r["kappa_D"] for r in results]).reshape(2, len(T_VALS)).T

    # Collect TQ variance for BOTH bit-widths
    tq_var_2 = np.array([r["tq_avg_var"][BITS[0]] for r in results]).reshape(2, len(T_VALS)).T
    tq_var_3 = np.array([r["tq_avg_var"][BITS[1]] for r in results]).reshape(2, len(T_VALS)).T

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Cosine similarity (changes with D)
    sns.heatmap(cos_data, ax=axes[0, 0], xticklabels=["Eq.1", "Eq.2"],
                yticklabels=[f"t={t}" for t in T_VALS], annot=True, fmt=".4f",
                cmap="RdYlBu_r", cbar_kws={"label": "Mean Cosine Similarity"})
    axes[0, 0].set_title("Cosine Similarity (Arbitrary under D)")
    axes[0, 0].set_ylabel("D Skew")

    # Panel 2: Raw dot product (invariant under D)
    sns.heatmap(raw_data, ax=axes[0, 1], xticklabels=["Eq.1", "Eq.2"],
                yticklabels=[f"t={t}" for t in T_VALS], annot=True, fmt=".4f",
                cmap="RdYlBu_r", cbar_kws={"label": "Mean Dot Product"})
    axes[0, 1].set_title("Raw Dot Product (D-Invariant)")
    axes[0, 1].set_ylabel("D Skew")

    # Panel 3: TQ variance (LOG SCALE) — 2-bit
    tq2_log = np.log10(tq_var_2 + 1e-10)
    sns.heatmap(tq2_log, ax=axes[1, 0], xticklabels=["Eq.1", "Eq.2"],
                yticklabels=[f"t={t}" for t in T_VALS], annot=True, fmt=".1f",
                cmap="YlOrRd", cbar_kws={"label": "log₁₀(Variance)"})
    axes[1, 0].set_title(f"TurboQuant {BITS[0]}-bit: Variance (log₁₀)")
    axes[1, 0].set_ylabel("D Skew")

    # Panel 4: TQ variance (LOG SCALE) — 3-bit
    tq3_log = np.log10(tq_var_3 + 1e-10)
    sns.heatmap(tq3_log, ax=axes[1, 1], xticklabels=["Eq.1", "Eq.2"],
                yticklabels=[f"t={t}" for t in T_VALS], annot=True, fmt=".1f",
                cmap="YlOrRd", cbar_kws={"label": "log₁₀(Variance)"})
    axes[1, 1].set_title(f"TurboQuant {BITS[1]}-bit: Variance (log₁₀)")
    axes[1, 1].set_ylabel("D Skew")

    plt.suptitle("Netflix D-Scaling Invariance × TurboQuant Quantization Variance\n"
                 "(MovieLens-1M, K=50, λ=0.01)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("/results/netflix_turboquant_heatmaps.png", dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Main Modal function
# ---------------------------------------------------------------------------

@app.function(image=image, volumes={"/results": volume}, timeout=3600)
def run_experiments():
    """Replicate Netflix heatmaps + add TurboQuant layer."""
    import torch
    import numpy as np
    from turboquant import TurboQuantIP

    # Load pre-downloaded data
    data = torch.load("/results/movielens_1m.pt", weights_only=False)
    X = data["X"]
    n_users, n_items = X.shape
    print(f"Loaded X: {n_users} users × {n_items} items")

    # Fixed z so κ(D) grows monotonically with t across all conditions
    torch.manual_seed(99)
    z = torch.randn(K)
    results = []

    for reg_scheme in ["eq1", "eq2"]:
        print(f"\nTraining MF ({reg_scheme})...")
        A, B = train_mf(X, reg_scheme)
        user_emb = X @ A   # n_users × K
        item_emb = B        # n_items × K

        for t in T_VALS:
            D, diags = generate_D(t, z)
            D_inv = torch.diag(1.0 / diags)

            # Netflix Eq.3 scaling
            user_emb_s = user_emb @ D       # u^(D) = uD
            item_emb_s = item_emb @ D_inv   # v^(D) = vD^{-1}

            # Fixed pair sample for all conditions
            np.random.seed(42)
            user_idxs = np.random.choice(n_users, N_PAIRS, replace=False)
            item_idxs = np.random.choice(n_items, N_PAIRS, replace=False)

            # 1. Cosine similarity (arbitrary under D scaling)
            cos_vals = []
            for u in user_idxs:
                for v in item_idxs:
                    uv, vv = user_emb_s[u], item_emb_s[v]
                    cos_vals.append((torch.dot(uv, vv) /
                                     (torch.norm(uv) * torch.norm(vv) + 1e-8)).item())

            # 2. Raw dot product (D-invariant)
            dot_vals = []
            for u in user_idxs:
                for v in item_idxs:
                    dot_vals.append(torch.dot(user_emb_s[u], item_emb_s[v]).item())

            # 3. TurboQuant: per-pair variance across different rotation seeds
            pair_keys = [(u, v) for u in user_idxs[:5] for v in item_idxs[:5]]
            tq_avg_var = {}

            for b in BITS:
                pair_dots = {k: [] for k in pair_keys}

                for seed in range(M_SEEDS):
                    # New TurboQuantIP per seed → different rotation Π and QJL S.
                    # quantize/dequantize is deterministic given these matrices.
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", DeprecationWarning)
                        tq = TurboQuantIP(dim=K, bits=b, device='cpu', seed=seed)

                    # quantize returns 4-tuple; dequantize takes same 4 args
                    mse_idx, norms, qjl_signs, res_norms = tq.quantize(item_emb_s)
                    item_dequant = tq.dequantize(mse_idx, norms, qjl_signs, res_norms)

                    for u, v in pair_keys:
                        dot = torch.dot(user_emb_s[u], item_dequant[v]).item()
                        pair_dots[(u, v)].append(dot)

                # Variance per pair across seeds, then average
                tq_avg_var[b] = float(np.mean([np.var(dots) for dots in pair_dots.values()]))

            results.append({
                "reg_scheme": reg_scheme,
                "t": t,
                "kappa_D": (torch.max(diags) / torch.min(diags)).item(),
                "cos_sim_mean": float(np.mean(cos_vals)),
                "cos_sim_var":  float(np.var(cos_vals)),
                "raw_dot_mean": float(np.mean(dot_vals)),
                "raw_dot_var":  float(np.var(dot_vals)),
                "tq_avg_var":   tq_avg_var,   # dict: {2: float, 3: float}
            })
            print(f"  t={t}, κ(D)={results[-1]['kappa_D']:.2f}, "
                  f"tq_var={tq_avg_var}")

    torch.save(results, "/results/experiment_results.pt")
    volume.commit()

    generate_heatmaps(results)
    volume.commit()

    print("\nExperiments complete. Results and heatmaps saved.")
    return "Done"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with app.run():
        print(run_experiments.remote())
