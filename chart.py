"""
Charting script — runs on Modal, reads experiment_data.pt from volume.
Generates:
  - Figure 1: Netflix replication (item-item cosine, 1 row)
  - Figure 2: TQ contribution (scatter + variance vs κ)
  - Table 1 data printed to stdout

No experiment re-running. Pure visualization.
"""
import modal

app = modal.App("tq-charts")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "matplotlib", "scipy")
    .add_local_file("turboquant_impl.py", "/root/turboquant_impl.py")
)
volume = modal.Volume.from_name("turboquant-netflix-results", create_if_missing=True)

# --- Config (must match experiment.py) ---
K = 50
N_ITEMS = 1000
LAMBDA_EQ1 = 10_000


@app.function(image=image, volumes={"/results": volume}, timeout=3600)
def generate_charts():
    import torch
    import numpy as np
    import sys
    sys.path.insert(0, "/root")
    from turboquant_impl import TurboQuantIP, MSEOnlyQuantizer
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator

    # --- Load data ---
    data = torch.load("/results/experiment_data.pt", map_location="cpu", weights_only=False)
    A_eq1 = data["A_eq1"]
    B_eq1 = data["B_eq1"]
    A_eq2 = data["A_eq2"]
    B_eq2 = data["B_eq2"]
    X = data["X"]
    item_clusters = data["item_clusters"]
    print("Loaded experiment_data.pt")

    # --- Helpers ---
    def sort_items_by_cluster(clusters):
        return np.argsort(clusters, kind='stable')

    def get_D_scalings(B, lam):
        U, sigma, Vt = torch.linalg.svd(B, full_matrices=False)
        scalings = {}
        scalings["B = V_k"] = torch.eye(K)
        scalings["B = V_k·dMat(σ²)"] = torch.diag(sigma ** 2)
        scalings["B = V_k·dMat((1+λ/σ²)^½)"] = torch.diag(
            torch.sqrt(1.0 + lam / (sigma ** 2 + 1e-10)))
        inner = sigma * torch.clamp(1.0 - lam / (sigma ** 2 + 1e-10), min=0.0)
        scalings["B = V_k·dMat(σ(1-λ/σ²)₊^½)"] = torch.diag(
            torch.sqrt(inner + 1e-10))
        return scalings

    item_sort = sort_items_by_cluster(item_clusters)
    scalings = get_D_scalings(B_eq1, LAMBDA_EQ1)

    # =====================================================================
    # FIGURE 1: Netflix replication (item-item cosine only)
    # =====================================================================
    print("\n--- Figure 1: Netflix replication ---")

    def cosine_sim_matrix(B_scaled):
        norms = torch.norm(B_scaled, dim=1, keepdim=True).clamp(min=1e-8)
        B_normed = B_scaled / norms
        return (B_normed @ B_normed.T).numpy()

    # True cluster matrix
    true_sim = np.zeros((N_ITEMS, N_ITEMS), dtype=np.float32)
    for i in range(N_ITEMS):
        for j in range(N_ITEMS):
            if item_clusters[i] == item_clusters[j]:
                true_sim[i, j] = 1.0

    scaling_names = list(scalings.keys())
    cos_matrices = {}
    for name, D in scalings.items():
        cos_matrices[name] = cosine_sim_matrix(B_eq1 @ D)

    cos_eq2 = cosine_sim_matrix(B_eq2)

    n_cols = 1 + len(scaling_names) + 1
    fig1, axes1 = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.2))

    def plot_item_item(ax, mat, title, cmap="RdBu_r", vmin=None, vmax=None):
        s = mat[np.ix_(item_sort, item_sort)]
        if vmin is None: vmin = np.percentile(s, 2)
        if vmax is None: vmax = np.percentile(s, 98)
        ax.imshow(s, cmap=cmap, vmin=vmin, vmax=vmax,
                  aspect='equal', interpolation='nearest')
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    plot_item_item(axes1[0], true_sim, "True Clusters", cmap="bone_r", vmin=0, vmax=1)
    for i, name in enumerate(scaling_names):
        plot_item_item(axes1[1 + i], cos_matrices[name], f"cosSim: {name}")
    plot_item_item(axes1[-1], cos_eq2, "cosSim: Eq.2 (ref)")

    fig1.suptitle(
        "Netflix Figure 1 Replication: Cosine Similarity Is Arbitrary Under D-Scaling\n"
        "(Synthetic: p=1,000 items, C=5 clusters, K=50, Eq.1 λ=10,000)",
        fontsize=10, fontweight="bold")
    fig1.tight_layout()
    fig1.savefig("/results/figure1_netflix.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)
    print("Saved figure1_netflix.png")

    # =====================================================================
    # FIGURE 2: TQ contribution — scatter + variance vs κ
    # =====================================================================
    print("\n--- Figure 2: TQ contribution ---")

    torch.manual_seed(99)
    z = torch.randn(K)
    U_full = X @ A_eq1  # (20000, K)

    # t values for D-scaling: D = diag(exp(t*z))
    t_vals = [0, 0.5, 1, 2]
    N_SEEDS = 100
    N_USERS_SAMPLE = 200
    N_ITEMS_SAMPLE = 50

    np.random.seed(42)
    user_idx = np.random.choice(20000, N_USERS_SAMPLE, replace=False)
    item_idx = np.random.choice(N_ITEMS, N_ITEMS_SAMPLE, replace=False)

    results = {}  # t -> {kappa, true_dots, tq_2bit, tq_3bit, mse_2bit, mse_3bit}

    for t in t_vals:
        diags = torch.exp(t * z)
        kappa = (torch.max(diags) / torch.min(diags)).item()
        D = torch.diag(diags)
        D_inv = torch.diag(1.0 / diags)

        U_s = U_full @ D        # users scaled
        V_s = B_eq1 @ D_inv     # items scaled

        # True dot products for sampled pairs
        U_sub = U_s[user_idx]   # (200, K)
        V_sub = V_s[item_idx]   # (50, K)
        true_dots = (U_sub @ V_sub.T).numpy().ravel()  # (10000,)

        # TQ estimates: collect per-seed for variance
        tq_estimates = {2: [], 3: []}
        mse_estimates = {2: [], 3: []}

        for bits in [2, 3]:
            for seed in range(N_SEEDS):
                # TurboQuant (b-1 bits MSE + 1 bit QJL = b bits total)
                tq = TurboQuantIP(dim=K, bits=bits, seed=seed)
                q = tq.quantize(V_s)
                V_deq = tq.dequantize(*q)
                V_deq_sub = V_deq[item_idx]
                est = (U_sub @ V_deq_sub.T).numpy().ravel()
                tq_estimates[bits].append(est)

                # Fair MSE-only baseline (all b bits for MSE, no QJL)
                mse_q = MSEOnlyQuantizer(dim=K, bits=bits, seed=seed)
                mse_idx, norms = mse_q.quantize(V_s)
                V_mse = mse_q.dequantize(mse_idx, norms)
                V_mse_sub = V_mse[item_idx]
                est_mse = (U_sub @ V_mse_sub.T).numpy().ravel()
                mse_estimates[bits].append(est_mse)

        results[t] = {
            "kappa": kappa,
            "true_dots": true_dots,
        }
        for bits in [2, 3]:
            arr_tq = np.array(tq_estimates[bits])     # (N_SEEDS, 10000)
            arr_mse = np.array(mse_estimates[bits])
            results[t][f"tq_{bits}bit_mean"] = arr_tq.mean(axis=0)
            results[t][f"tq_{bits}bit_var"] = arr_tq.var(axis=0)
            results[t][f"tq_{bits}bit_all"] = arr_tq
            results[t][f"mse_{bits}bit_mean"] = arr_mse.mean(axis=0)
            results[t][f"mse_{bits}bit_var"] = arr_mse.var(axis=0)

        print(f"  t={t}, κ={kappa:.1f}")

    # --- Figure 2A: 3×4 Scatter (transposed: rows=κ, cols=method) ---
    fig2, axes2 = plt.subplots(3, 4, figsize=(10, 7.5))

    scatter_ts = [0, 0.5, 1]
    col_configs = [
        ("tq_3bit_mean",  "TQ 3-bit",      "steelblue"),
        ("tq_2bit_mean",  "TQ 2-bit",      "cornflowerblue"),
        ("mse_3bit_mean", "MSE-only 3-bit", "indianred"),
        ("mse_2bit_mean", "MSE-only 2-bit", "lightsalmon"),
    ]

    for row, t in enumerate(scatter_ts):
        r = results[t]
        true = r["true_dots"]

        for col, (key, label, color) in enumerate(col_configs):
            ax = axes2[row, col]
            est = r[key]
            bias = (est - true).mean()

            ax.scatter(true, est, s=0.3, alpha=0.3, c=color, rasterized=True)
            ax.plot([0, 1], [0, 1], 'k-', lw=0.8, alpha=0.5)
            ax.set_title(f"{label} | κ={r['kappa']:.0f}\nbias={bias:.4f}", fontsize=8)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.tick_params(labelsize=6)

            if row == 0:
                pass  # title already has method name
            if col == 0:
                ax.set_ylabel(f"κ={r['kappa']:.0f}\nestimate", fontsize=8)
            if row == 2:
                ax.set_xlabel("True ⟨u, v⟩", fontsize=7)

    fig2.suptitle(
        "TurboQuant vs MSE-Only Inner Product Estimates\n"
        "Each dot = one (user, item) pair, MC mean over 100 seeds",
        fontsize=11, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig("/results/figure2a_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print("Saved figure2a_scatter.png")

    # --- Figure 2B: Variance vs κ(D) ---
    fig3, (ax_bias, ax_var) = plt.subplots(1, 2, figsize=(10, 4.5))

    kappas = [results[t]["kappa"] for t in t_vals]

    for bits, marker in [(2, 's'), (3, 'o')]:
        # TQ bias with SEM error bars
        biases_tq = []
        sems_tq = []
        vars_tq = []
        biases_mse = []
        vars_mse = []
        for t in t_vals:
            r = results[t]
            true = r["true_dots"]
            tq_mean = r[f"tq_{bits}bit_mean"]
            mse_mean = r[f"mse_{bits}bit_mean"]

            pair_bias = tq_mean - true
            biases_tq.append(pair_bias.mean())
            sems_tq.append(pair_bias.std() / np.sqrt(len(pair_bias)))
            vars_tq.append(r[f"tq_{bits}bit_var"].mean())

            pair_bias_mse = mse_mean - true
            biases_mse.append(pair_bias_mse.mean())
            vars_mse.append(r[f"mse_{bits}bit_var"].mean())

        # Bias panel
        ax_bias.errorbar(kappas, biases_tq, yerr=[1.96*s for s in sems_tq],
                         fmt=f'-{marker}', label=f"TQ {bits}-bit", capsize=3,
                         color="steelblue" if bits == 3 else "cornflowerblue")
        ax_bias.plot(kappas, biases_mse, f'--{marker}',
                     label=f"MSE-only {bits}-bit",
                     color="indianred" if bits == 3 else "lightsalmon")

        # Variance panel
        ax_var.plot(kappas, vars_tq, f'-{marker}', label=f"TQ {bits}-bit",
                    color="steelblue" if bits == 3 else "cornflowerblue")
        ax_var.plot(kappas, vars_mse, f'--{marker}', label=f"MSE-only {bits}-bit",
                    color="indianred" if bits == 3 else "lightsalmon")

    ax_bias.axhline(0, color='k', lw=0.5, ls=':')
    ax_bias.set_xscale('log')
    ax_bias.set_xlabel("κ(D)")
    ax_bias.set_ylabel("Mean Bias (with 95% CI)")
    ax_bias.set_title("Bias vs Condition Number")
    ax_bias.legend(fontsize=7)

    ax_var.set_xscale('log')
    ax_var.set_yscale('log')
    ax_var.set_xlabel("κ(D)")
    ax_var.set_ylabel("Mean Variance")
    ax_var.set_title("Variance vs Condition Number")
    ax_var.legend(fontsize=7)

    fig3.suptitle(
        "Figure 2B: Bias and Variance Under D-Scaling\n"
        "User-item scores ⟨u⁽ᴰ⁾, ṽ⁽ᴰ⁾⟩, 100 seeds, 10K pairs per κ",
        fontsize=11, fontweight="bold")
    fig3.tight_layout()
    fig3.savefig("/results/figure2b_bias_variance.png", dpi=200, bbox_inches="tight")
    plt.close(fig3)
    print("Saved figure2b_bias_variance.png")

    # =====================================================================
    # TABLE 1: Print quantitative results
    # =====================================================================
    print("\n--- Table 1: Quantitative Results ---")
    print(f"{'t':>4} {'κ(D)':>12} {'True ⟨u,v⟩':>12} | "
          f"{'TQ-3b Mean':>10} {'Bias':>10} {'±SEM':>8} {'Var':>10} | "
          f"{'MSE-3b Mean':>11} {'Bias':>10} {'Var':>10}")
    print("-" * 120)

    for t in t_vals:
        r = results[t]
        true = r["true_dots"]
        tq3 = r["tq_3bit_mean"]
        mse3 = r["mse_3bit_mean"]

        true_mean = true.mean()
        tq_bias = (tq3 - true).mean()
        tq_sem = (tq3 - true).std() / np.sqrt(len(true))
        tq_var = r["tq_3bit_var"].mean()
        mse_bias = (mse3 - true).mean()
        mse_var = r["mse_3bit_var"].mean()

        print(f"{t:>4} {r['kappa']:>12.1f} {true_mean:>12.6f} | "
              f"{tq3.mean():>10.6f} {tq_bias:>+10.6f} {tq_sem:>8.6f} {tq_var:>10.4e} | "
              f"{mse3.mean():>11.6f} {mse_bias:>+10.6f} {mse_var:>10.4e}")

    print("\n--- Table 1 (2-bit) ---")
    for t in t_vals:
        r = results[t]
        true = r["true_dots"]
        tq2 = r["tq_2bit_mean"]
        mse2 = r["mse_2bit_mean"]

        true_mean = true.mean()
        tq_bias = (tq2 - true).mean()
        tq_sem = (tq2 - true).std() / np.sqrt(len(true))
        tq_var = r["tq_2bit_var"].mean()
        mse_bias = (mse2 - true).mean()
        mse_var = r["mse_2bit_var"].mean()

        print(f"{t:>4} {r['kappa']:>12.1f} {true_mean:>12.6f} | "
              f"{tq2.mean():>10.6f} {tq_bias:>+10.6f} {tq_sem:>8.6f} {tq_var:>10.4e} | "
              f"{mse2.mean():>11.6f} {mse_bias:>+10.6f} {mse_var:>10.4e}")

    print("\nAll charts generated.")


if __name__ == "__main__":
    with app.run():
        print(generate_charts.remote())
