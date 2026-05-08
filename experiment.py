"""
TurboQuant × Netflix: Synthetic Experiment
Replicates Netflix Section 4 setup exactly, then adds TurboQuant quantized
dot product to show cluster structure is preserved under compression.

Run: `modal run experiment.py`
"""
import modal

# --- Modal setup ---
app = modal.App("turboquant-netflix-experiments")
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "numpy",
        "matplotlib",
        "seaborn",
        "scipy",
    )
    .add_local_file("turboquant_impl.py", "/root/turboquant_impl.py")
)
volume = modal.Volume.from_name("turboquant-netflix-results", create_if_missing=True)

# --- Configuration (matching Netflix Section 4) ---
N_USERS = 20_000
N_ITEMS = 1_000
C = 5                    # number of clusters
K = 50                   # latent rank
BETA_ITEM_MIN = 0.25     # power-law exponent range for item popularity
BETA_ITEM_MAX = 1.5
BETA_USER = 0.5          # power-law exponent for user activity
LAMBDA_EQ1 = 10_000      # Netflix's λ for Eq.1
LAMBDA_EQ2 = 100          # Netflix's λ for Eq.2
BITS = [2, 3]             # TurboQuant bit-widths
M_SEEDS = 100             # rotation seeds for variance/bias estimation
N_EVAL_PAIRS = 200        # user-item pairs for quantitative metrics


# ---------------------------------------------------------------------------
# Synthetic data generation (Netflix Section 4)
# ---------------------------------------------------------------------------

def generate_synthetic_data(seed=42):
    """Generate Netflix-style synthetic interaction matrix with known clusters."""
    import numpy as np

    rng = np.random.RandomState(seed)

    # Assign each item to one of C clusters uniformly
    item_clusters = rng.randint(0, C, size=N_ITEMS)

    # Sample power-law exponent per cluster, then assign item popularity
    beta_c = rng.uniform(BETA_ITEM_MIN, BETA_ITEM_MAX, size=C)
    item_popularity = np.zeros(N_ITEMS)
    for i in range(N_ITEMS):
        # Power-law: p_i ~ PowerLaw(beta_c[cluster_i])
        # Using inverse CDF: U ~ Uniform(0,1), X = U^(-1/beta)
        u = rng.uniform(0.01, 1.0)  # avoid zero
        item_popularity[i] = u ** (-1.0 / beta_c[item_clusters[i]])
    # Normalize per cluster so probabilities are well-scaled
    item_popularity /= item_popularity.sum()

    # Sample user-cluster preferences (how much each user likes each cluster)
    # p_uc ~ Dirichlet-like: just sample positive values and normalize
    user_cluster_pref = rng.dirichlet(np.ones(C), size=N_USERS)

    # Compute user-item interaction probabilities
    # p_ui = (p_{u,c_i} * p_i) / sum_j(p_{u,c_j} * p_j)
    # This gives each user a preference-weighted probability over items
    p_ui = np.zeros((N_USERS, N_ITEMS))
    for u in range(N_USERS):
        for i in range(N_ITEMS):
            p_ui[u, i] = user_cluster_pref[u, item_clusters[i]] * item_popularity[i]
        p_ui[u] /= p_ui[u].sum()

    # Sample number of interactions per user: k_u ~ PowerLaw(beta_user)
    k_users = np.clip(
        (rng.uniform(0.01, 1.0, size=N_USERS) ** (-1.0 / BETA_USER)).astype(int),
        1, N_ITEMS // 2
    )

    # Generate binary interaction matrix
    X = np.zeros((N_USERS, N_ITEMS), dtype=np.float32)
    for u in range(N_USERS):
        chosen = rng.choice(N_ITEMS, size=min(k_users[u], N_ITEMS),
                            replace=False, p=p_ui[u])
        X[u, chosen] = 1.0

    return X, item_clusters


def sort_items_by_cluster(item_clusters):
    """Return sort order: by cluster, then by index within cluster."""
    import numpy as np
    return np.argsort(item_clusters, kind='stable')


# ---------------------------------------------------------------------------
# Matrix factorization (Netflix Eq.1 and Eq.2)
# ---------------------------------------------------------------------------

def train_mf(X_torch, reg_scheme="eq1", lam=None):
    """Train linear MF: X ≈ X A B^T with Netflix regularization."""
    import torch

    if lam is None:
        lam = LAMBDA_EQ1 if reg_scheme == "eq1" else LAMBDA_EQ2

    torch.manual_seed(0)
    n, p = X_torch.shape
    A = torch.nn.Parameter(torch.randn(p, K) * 0.01)
    B = torch.nn.Parameter(torch.randn(p, K) * 0.01)
    optimizer = torch.optim.Adam([A, B], lr=1e-3)

    for epoch in range(200):
        optimizer.zero_grad()
        recon = X_torch @ A @ B.T
        loss = ((X_torch - recon) ** 2).sum()
        if reg_scheme == "eq1":
            loss += lam * ((A @ B.T) ** 2).sum()
        else:  # eq2
            loss += lam * ((X_torch @ A) ** 2).sum() + lam * (B ** 2).sum()
        loss.backward()
        optimizer.step()

    return A.detach(), B.detach()


# ---------------------------------------------------------------------------
# D-scaling variants (Netflix Section 2.2)
# ---------------------------------------------------------------------------

def get_D_scalings(B, lam):
    """Return dict of named D-scalings matching Netflix Figure 1.
    Netflix uses specific scalings based on singular values of V_k."""
    import torch

    # SVD of B to get singular values σ_i
    U, sigma, Vt = torch.linalg.svd(B, full_matrices=False)

    scalings = {}

    # 1. Identity (no rescaling): B = V_k
    scalings["B = V_k"] = torch.eye(K)

    # 2. B = V_k * dMat(σ_i^2): scale by squared singular values
    scalings["B = V_k·dMat(σ²)"] = torch.diag(sigma ** 2)

    # 3. B = V_k * dMat((1 + λ/σ²)^{1/2}): Netflix's specific scaling
    scalings["B = V_k·dMat((1+λ/σ²)^½)"] = torch.diag(
        torch.sqrt(1.0 + lam / (sigma ** 2 + 1e-10))
    )

    # 4. B = V_k * dMat(σ(1-λ/σ²)_+^{1/2}): Netflix's other scaling
    inner = sigma * torch.clamp(1.0 - lam / (sigma ** 2 + 1e-10), min=0.0)
    scalings["B = V_k·dMat(σ(1-λ/σ²)₊^½)"] = torch.diag(
        torch.sqrt(inner + 1e-10)
    )

    return scalings


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def generate_figure1(B_eq1, B_eq2, item_clusters, lam_eq1):
    """
    Replicate Netflix Figure 1 + add TurboQuant row.
    Row 1: True clusters | cosSim under 3 D-scalings (Eq.1) | cosSim Eq.2
    Row 2: TQ dot product under same D-scalings | TQ dot product Eq.2
    """
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, "/root")
    from turboquant_impl import TurboQuantIP

    sort_idx = sort_items_by_cluster(item_clusters)

    # --- Ground truth cluster similarity matrix ---
    true_sim = np.zeros((N_ITEMS, N_ITEMS), dtype=np.float32)
    for i in range(N_ITEMS):
        for j in range(N_ITEMS):
            if item_clusters[i] == item_clusters[j]:
                true_sim[i, j] = 1.0
    true_sim_sorted = true_sim[np.ix_(sort_idx, sort_idx)]

    # --- D-scalings for Eq.1 ---
    scalings = get_D_scalings(B_eq1, lam_eq1)

    def cosine_sim_matrix(B_scaled):
        """Compute p×p item-item cosine similarity matrix."""
        norms = torch.norm(B_scaled, dim=1, keepdim=True).clamp(min=1e-8)
        B_normed = B_scaled / norms
        return (B_normed @ B_normed.T).numpy()

    def dot_product_matrix(B_scaled):
        """Compute p×p item-item dot product matrix."""
        return (B_scaled @ B_scaled.T).numpy()

    def tq_dot_product_matrix(B_scaled, bits=3, n_seeds=10):
        """Compute p×p TQ-quantized dot product matrix, averaged over seeds."""
        accum = np.zeros((N_ITEMS, N_ITEMS), dtype=np.float64)
        for seed in range(n_seeds):
            tq = TurboQuantIP(dim=K, bits=bits, seed=seed)
            mse_idx, norms, qjl_signs, res_norms = tq.quantize(B_scaled)
            B_deq = tq.dequantize(mse_idx, norms, qjl_signs, res_norms)
            # One-sided: use dequantized items as both arguments for item-item
            # This shows whether the *structure* is preserved, not exact values
            accum += (B_deq @ B_deq.T).numpy()
        return accum / n_seeds

    # Compute all cosine similarity matrices (Eq.1 D-scalings)
    scaling_names = list(scalings.keys())
    cos_matrices = {}
    tq_matrices = {}
    for name, D in scalings.items():
        B_scaled = B_eq1 @ D
        cos_matrices[name] = cosine_sim_matrix(B_scaled)
        tq_matrices[name] = tq_dot_product_matrix(B_scaled, bits=3, n_seeds=20)

    # Eq.2 (unique solution)
    cos_eq2 = cosine_sim_matrix(B_eq2)
    tq_eq2 = tq_dot_product_matrix(B_eq2, bits=3, n_seeds=20)
    dot_eq2 = dot_product_matrix(B_eq2)

    # --- Build figure ---
    n_cols = 1 + len(scaling_names) + 1  # true + scalings + eq2
    fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 11))

    def plot_matrix(ax, mat, title, cmap="RdBu_r", vmin=None, vmax=None):
        sorted_mat = mat[np.ix_(sort_idx, sort_idx)]
        if vmin is None:
            vmin = np.percentile(sorted_mat, 2)
        if vmax is None:
            vmax = np.percentile(sorted_mat, 98)
        ax.imshow(sorted_mat, cmap=cmap, vmin=vmin, vmax=vmax,
                  aspect='equal', interpolation='nearest')
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # Row 0: Ground truth + Cosine similarities (Netflix replication)
    plot_matrix(axes[0, 0], true_sim, "True Clusters", cmap="bone_r", vmin=0, vmax=1)
    for i, name in enumerate(scaling_names):
        plot_matrix(axes[0, 1 + i], cos_matrices[name], f"cosSim: {name}")
    plot_matrix(axes[0, -1], cos_eq2, "cosSim: Eq.2 (unique)")

    # Row 1: Raw dot products under same scalings (should all look the same)
    plot_matrix(axes[1, 0], true_sim, "True Clusters", cmap="bone_r", vmin=0, vmax=1)
    for i, name in enumerate(scaling_names):
        B_scaled = B_eq1 @ scalings[name]
        dot_mat = dot_product_matrix(B_scaled)
        plot_matrix(axes[1, 1 + i], dot_mat, f"dotProd: {name}")
    plot_matrix(axes[1, -1], dot_eq2, "dotProd: Eq.2")

    # Row 2: TurboQuant quantized dot products (our contribution)
    plot_matrix(axes[2, 0], true_sim, "True Clusters", cmap="bone_r", vmin=0, vmax=1)
    for i, name in enumerate(scaling_names):
        plot_matrix(axes[2, 1 + i], tq_matrices[name], f"TQ 3-bit: {name}")
    plot_matrix(axes[2, -1], tq_eq2, "TQ 3-bit: Eq.2")

    # Row labels
    axes[0, 0].set_ylabel("Cosine Similarity\n(Netflix Fig.1)", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Raw Dot Product\n(D-Invariant)", fontsize=11, fontweight="bold")
    axes[2, 0].set_ylabel("TurboQuant 3-bit\n(Ours)", fontsize=11, fontweight="bold")

    plt.suptitle(
        "Netflix D-Scaling Arbitrariness × TurboQuant Quantized Dot Products\n"
        f"(Synthetic: n={N_USERS:,}, p={N_ITEMS:,}, C={C}, K={K})",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig("/results/figure1_replication.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved figure1_replication.png")


# ---------------------------------------------------------------------------
# Quantitative table: bias, variance, unbiasedness verification
# ---------------------------------------------------------------------------

def compute_quantitative_results(A_eq1, B_eq1, X_torch):
    """
    For Eq.1 only (D-scaling is meaningful here), compute:
    - Mean cosine similarity across D-scalings
    - Mean dot product (should be D-invariant)
    - TQ mean estimate vs true dot product (bias verification)
    - TQ variance across seeds
    """
    import torch
    import numpy as np
    import sys
    sys.path.insert(0, "/root")
    from turboquant_impl import TurboQuantIP

    torch.manual_seed(99)
    z = torch.randn(K)

    t_vals = [0, 0.5, 1, 2, 5]

    # Sample evaluation pairs
    np.random.seed(42)
    user_idxs = np.random.choice(N_USERS, min(N_EVAL_PAIRS, N_USERS), replace=False)
    item_idxs = np.random.choice(N_ITEMS, min(N_EVAL_PAIRS, N_ITEMS), replace=False)

    user_emb = X_torch @ A_eq1
    item_emb = B_eq1

    results = []

    for t in t_vals:
        diags = torch.exp(t * z)
        D = torch.diag(diags)
        D_inv = torch.diag(1.0 / diags)
        kappa = (torch.max(diags) / torch.min(diags)).item()

        user_emb_s = user_emb @ D
        item_emb_s = item_emb @ D_inv

        # 1. Cosine similarity
        cos_vals = []
        for u in user_idxs:
            for v in item_idxs[:20]:  # 200 * 20 = 4000 pairs
                uv, vv = user_emb_s[u], item_emb_s[v]
                c = (torch.dot(uv, vv) / (torch.norm(uv) * torch.norm(vv) + 1e-8)).item()
                cos_vals.append(c)

        # 2. Raw dot product
        dot_vals = []
        for u in user_idxs:
            for v in item_idxs[:20]:
                dot_vals.append(torch.dot(user_emb_s[u], item_emb_s[v]).item())

        # 3. TurboQuant: bias and variance verification
        tq_results = {}
        for b in BITS:
            pair_dots = []  # list of lists: [seed1_dots, seed2_dots, ...]

            for seed in range(M_SEEDS):
                tq = TurboQuantIP(dim=K, bits=b, seed=seed)

                mse_idx, norms, qjl_signs, res_norms = tq.quantize(item_emb_s)
                item_deq = tq.dequantize(mse_idx, norms, qjl_signs, res_norms)

                seed_dots = []
                for u in user_idxs[:50]:
                    for v in item_idxs[:20]:
                        seed_dots.append(torch.dot(user_emb_s[u], item_deq[v]).item())
                pair_dots.append(seed_dots)

            pair_dots = np.array(pair_dots)  # (M_SEEDS, n_pairs)

            # True dot products for same pairs
            true_dots = []
            for u in user_idxs[:50]:
                for v in item_idxs[:20]:
                    true_dots.append(torch.dot(user_emb_s[u], item_emb_s[v]).item())
            true_dots = np.array(true_dots)

            # Bias: E[TQ estimate] - true value, averaged over pairs
            mean_tq = pair_dots.mean(axis=0)  # mean over seeds per pair
            bias = (mean_tq - true_dots).mean()
            mean_abs_bias = np.abs(mean_tq - true_dots).mean()

            # Variance: across seeds, averaged over pairs
            var_per_pair = pair_dots.var(axis=0)
            mean_var = var_per_pair.mean()

            tq_results[b] = {
                "bias": float(bias),
                "mean_abs_bias": float(mean_abs_bias),
                "mean_var": float(mean_var),
                "mean_true_dot": float(true_dots.mean()),
                "mean_tq_dot": float(mean_tq.mean()),
            }

        results.append({
            "t": t,
            "kappa_D": kappa,
            "cos_sim_mean": float(np.mean(cos_vals)),
            "raw_dot_mean": float(np.mean(dot_vals)),
            "tq": tq_results,
        })

        print(f"  t={t}, κ(D)={kappa:.2f}")
        for b in BITS:
            r = tq_results[b]
            print(f"    {b}-bit: bias={r['bias']:.6f}, |bias|={r['mean_abs_bias']:.6f}, "
                  f"var={r['mean_var']:.4e}, true_dot={r['mean_true_dot']:.6f}, "
                  f"tq_dot={r['mean_tq_dot']:.6f}")

    return results


# ---------------------------------------------------------------------------
# Main Modal function
# ---------------------------------------------------------------------------

@app.function(image=image, volumes={"/results": volume}, timeout=7200)
def run_experiments():
    """Full experiment: synthetic data + Netflix Figure 1 + TurboQuant extension."""
    import torch
    import numpy as np

    print("Generating synthetic data (Netflix Section 4)...")
    X_np, item_clusters = generate_synthetic_data(seed=42)
    X_torch = torch.tensor(X_np)
    print(f"X: {X_torch.shape}, clusters: {np.bincount(item_clusters)}")

    print("\nTraining MF (Eq.1, λ={})...".format(LAMBDA_EQ1))
    A_eq1, B_eq1 = train_mf(X_torch, "eq1", LAMBDA_EQ1)
    print(f"  A: {A_eq1.shape}, B: {B_eq1.shape}")

    print("\nTraining MF (Eq.2, λ={})...".format(LAMBDA_EQ2))
    A_eq2, B_eq2 = train_mf(X_torch, "eq2", LAMBDA_EQ2)
    print(f"  A: {A_eq2.shape}, B: {B_eq2.shape}")

    print("\nGenerating Figure 1 (Netflix replication + TQ extension)...")
    generate_figure1(B_eq1, B_eq2, item_clusters, LAMBDA_EQ1)
    volume.commit()

    print("\nComputing quantitative results (Eq.1 only, D-scaling)...")
    quant_results = compute_quantitative_results(A_eq1, B_eq1, X_torch)
    torch.save({
        "quant_results": quant_results,
        "item_clusters": item_clusters,
    }, "/results/experiment_results.pt")
    volume.commit()

    print("\nAll experiments complete.")
    return "Done"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with app.run():
        print(run_experiments.remote())
