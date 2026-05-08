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

    return X, item_clusters, user_cluster_pref


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

N_SAMPLE_USERS = 300  # users to show in user-item heatmap
TQ_FIGURE_SEEDS = 20  # seeds for Monte Carlo mean in figure


def generate_figure1(A_eq1, B_eq1, A_eq2, B_eq2, X_torch, item_clusters, user_cluster_pref, lam_eq1):
    """
    Three-row figure:
    Row 1: Item-item cosine similarity under D-scalings (Netflix Figure 1 replication)
    Row 2: User-item score matrix U^(D)·V^(D)^T — the TRUE D-invariant quantity
    Row 3: TQ compressed user-item scores U^(D)·Ṽ^(D)^T — averaged over seeds
    """
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import sys
    sys.path.insert(0, "/root")
    from turboquant_impl import TurboQuantIP

    item_sort = sort_items_by_cluster(item_clusters)

    # --- Select and sort users by dominant cluster preference ---
    dominant_cluster = np.argmax(user_cluster_pref, axis=1)  # (N_USERS,)
    # Sample N_SAMPLE_USERS users, stratified by cluster
    rng = np.random.RandomState(123)
    selected_users = []
    per_cluster = N_SAMPLE_USERS // C
    for c in range(C):
        candidates = np.where(dominant_cluster == c)[0]
        chosen = rng.choice(candidates, min(per_cluster, len(candidates)), replace=False)
        selected_users.append(chosen)
    selected_users = np.concatenate(selected_users)
    # Sort by dominant cluster, then by preference strength within cluster
    user_sort_keys = [(dominant_cluster[u], -user_cluster_pref[u, dominant_cluster[u]]) for u in selected_users]
    user_order = sorted(range(len(selected_users)), key=lambda i: user_sort_keys[i])
    user_idx = selected_users[user_order]  # final sorted user indices
    n_users_show = len(user_idx)

    # --- Ground truth block matrix (for reference column) ---
    true_block = np.zeros((n_users_show, N_ITEMS), dtype=np.float32)
    for ui, u in enumerate(user_idx):
        for vi, v in enumerate(range(N_ITEMS)):
            if dominant_cluster[u] == item_clusters[v]:
                true_block[ui, vi] = 1.0
    true_block_sorted = true_block[:, item_sort]

    # --- D-scalings for Eq.1 ---
    scalings = get_D_scalings(B_eq1, lam_eq1)
    scaling_names = list(scalings.keys())

    # --- Precompute all user embeddings (full precision) ---
    U_full = X_torch @ A_eq1  # (N_USERS, K)

    def cosine_sim_matrix(B_scaled):
        """Compute p×p item-item cosine similarity matrix."""
        norms = torch.norm(B_scaled, dim=1, keepdim=True).clamp(min=1e-8)
        B_normed = B_scaled / norms
        return (B_normed @ B_normed.T).numpy()

    def user_item_score_matrix(U_scaled, V_scaled):
        """Compute user-item score matrix for selected users."""
        scores = (U_scaled[user_idx] @ V_scaled.T).numpy()  # (n_users_show, N_ITEMS)
        return scores

    def tq_user_item_score_matrix(U_scaled, V_scaled, bits=3, n_seeds=TQ_FIGURE_SEEDS):
        """One-sided TQ: quantize items, dot with full-precision users. MC mean."""
        accum = np.zeros((n_users_show, N_ITEMS), dtype=np.float64)
        for seed in range(n_seeds):
            tq = TurboQuantIP(dim=K, bits=bits, seed=seed)
            q = tq.quantize(V_scaled)
            V_deq = tq.dequantize(*q)
            accum += (U_scaled[user_idx] @ V_deq.T).numpy()
        return accum / n_seeds

    # --- Compute matrices for each D-scaling ---
    cos_matrices = {}
    score_matrices = {}
    tq_score_matrices = {}

    for name, M in scalings.items():
        M_inv = torch.diag(1.0 / torch.diag(M).clamp(min=1e-10))
        B_scaled = B_eq1 @ M        # items: B·M (the scaling returned by get_D_scalings)
        U_scaled = U_full @ M_inv   # users: U·M^{-1} so that ⟨U·M^{-1}, B·M⟩ = ⟨U, B⟩

        cos_matrices[name] = cosine_sim_matrix(B_scaled)
        score_matrices[name] = user_item_score_matrix(U_scaled, B_scaled)
        tq_score_matrices[name] = tq_user_item_score_matrix(U_scaled, B_scaled, bits=3)
        print(f"  Computed D-scaling: {name}")

    # Eq.2 (unique solution, no D-scaling)
    U_eq2 = X_torch @ A_eq2
    cos_eq2 = cosine_sim_matrix(B_eq2)
    score_eq2 = user_item_score_matrix(U_eq2, B_eq2)
    tq_score_eq2 = tq_user_item_score_matrix(U_eq2, B_eq2, bits=3)
    print("  Computed Eq.2")

    # --- Compute shared color scales per row ---
    # Row 2 (scores): all D-scalings should be identical, use global range
    all_scores = [score_matrices[n][:, item_sort] for n in scaling_names]
    all_scores.append(score_eq2[:, item_sort])
    score_vmin = np.percentile(np.concatenate([s.ravel() for s in all_scores]), 2)
    score_vmax = np.percentile(np.concatenate([s.ravel() for s in all_scores]), 98)

    # Row 3 (TQ scores): same scale as Row 2 for direct comparison
    tq_vmin, tq_vmax = score_vmin, score_vmax

    # --- Build figure ---
    n_cols = 1 + len(scaling_names) + 1  # reference + scalings + eq2
    fig, axes = plt.subplots(3, n_cols, figsize=(3.5 * n_cols, 10))

    def plot_item_item(ax, mat, title, cmap="RdBu_r", vmin=None, vmax=None):
        sorted_mat = mat[np.ix_(item_sort, item_sort)]
        if vmin is None:
            vmin = np.percentile(sorted_mat, 2)
        if vmax is None:
            vmax = np.percentile(sorted_mat, 98)
        ax.imshow(sorted_mat, cmap=cmap, vmin=vmin, vmax=vmax,
                  aspect='equal', interpolation='nearest')
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    def plot_user_item(ax, mat, title, cmap="RdBu_r", vmin=None, vmax=None):
        sorted_mat = mat[:, item_sort]  # users already sorted, sort items
        im = ax.imshow(sorted_mat, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect='auto', interpolation='nearest')
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        return im

    # --- Row 0: Item-item cosine similarity (Netflix replication) ---
    true_sim = np.zeros((N_ITEMS, N_ITEMS), dtype=np.float32)
    for i in range(N_ITEMS):
        for j in range(N_ITEMS):
            if item_clusters[i] == item_clusters[j]:
                true_sim[i, j] = 1.0
    plot_item_item(axes[0, 0], true_sim, "True Item Clusters", cmap="bone_r", vmin=0, vmax=1)
    for i, name in enumerate(scaling_names):
        plot_item_item(axes[0, 1 + i], cos_matrices[name], f"cosSim: {name}")
    plot_item_item(axes[0, -1], cos_eq2, "cosSim: Eq.2 (ref)")

    # --- Row 1: User-item scores (D-invariant) ---
    plot_user_item(axes[1, 0], true_block, "True User-Item Blocks",
                   cmap="bone_r", vmin=0, vmax=1)
    for i, name in enumerate(scaling_names):
        plot_user_item(axes[1, 1 + i], score_matrices[name],
                       f"⟨u⁽ᴰ⁾, v⁽ᴰ⁾⟩: {name}",
                       vmin=score_vmin, vmax=score_vmax)
    plot_user_item(axes[1, -1], score_eq2, "⟨u, v⟩: Eq.2 (ref)",
                   vmin=score_vmin, vmax=score_vmax)

    # --- Row 2: TQ compressed user-item scores ---
    plot_user_item(axes[2, 0], true_block, "True User-Item Blocks",
                   cmap="bone_r", vmin=0, vmax=1)
    for i, name in enumerate(scaling_names):
        plot_user_item(axes[2, 1 + i], tq_score_matrices[name],
                       f"TQ 3-bit: {name}",
                       vmin=tq_vmin, vmax=tq_vmax)
    plot_user_item(axes[2, -1], tq_score_eq2, "TQ 3-bit: Eq.2 (ref)",
                   vmin=tq_vmin, vmax=tq_vmax)

    # --- Row labels ---
    axes[0, 0].set_ylabel("Cosine Similarity\n(Netflix Fig.1)",
                          fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("User-Item Scores\n(D-Invariant)",
                          fontsize=10, fontweight="bold")
    axes[2, 0].set_ylabel(f"TQ 3-bit Compressed\n(MC mean, {TQ_FIGURE_SEEDS} seeds)",
                          fontsize=10, fontweight="bold")

    plt.suptitle(
        "Netflix D-Scaling Arbitrariness × TurboQuant Compressed Scores\n"
        f"(Synthetic: n={N_USERS:,}, p={N_ITEMS:,}, C={C}, K={K}; "
        f"{n_users_show} users shown, sorted by cluster)",
        fontsize=12, fontweight="bold"
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
    X_np, item_clusters, user_cluster_pref = generate_synthetic_data(seed=42)
    X_torch = torch.tensor(X_np)
    print(f"X: {X_torch.shape}, clusters: {np.bincount(item_clusters)}")

    print("\nTraining MF (Eq.1, λ={})...".format(LAMBDA_EQ1))
    A_eq1, B_eq1 = train_mf(X_torch, "eq1", LAMBDA_EQ1)
    print(f"  A: {A_eq1.shape}, B: {B_eq1.shape}")

    print("\nTraining MF (Eq.2, λ={})...".format(LAMBDA_EQ2))
    A_eq2, B_eq2 = train_mf(X_torch, "eq2", LAMBDA_EQ2)
    print(f"  A: {A_eq2.shape}, B: {B_eq2.shape}")

    print("\nGenerating Figure 1 (Netflix replication + TQ extension)...")
    generate_figure1(A_eq1, B_eq1, A_eq2, B_eq2, X_torch, item_clusters, user_cluster_pref, LAMBDA_EQ1)
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
