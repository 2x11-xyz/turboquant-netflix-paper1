"""
Figure 3 alternative: 2D density contour plot for MSE vs TQ scatter.
Much easier to read than scatter with overplotting.

Run: `modal run generate_fig3_contour.py`
"""
import modal

app = modal.App("tq-fig3-contour")
volume = modal.Volume.from_name("turboquant-netflix-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scipy", "matplotlib")
    .add_local_file("turboquant_impl.py", "/root/turboquant_impl.py")
)


@app.function(image=image, volumes={"/results": volume}, timeout=300)
def generate_contour_figure():
    import sys
    sys.path.insert(0, "/root")
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from turboquant_impl import TurboQuantIP, MSEOnlyQuantizer

    # Generate data
    d = 512
    bits = 2
    torch.manual_seed(42)
    np.random.seed(42)
    n_items = 5000
    n_queries = 200

    items = torch.randn(n_items, d)
    items = items / items.norm(dim=1, keepdim=True)
    queries = torch.randn(n_queries, d)
    queries = queries / queries.norm(dim=1, keepdim=True)

    true_ip = (queries @ items.T).numpy()

    seed = 0
    tq = TurboQuantIP(dim=d, bits=bits, seed=seed)
    mse_q = MSEOnlyQuantizer(dim=d, bits=bits, seed=seed)

    items_hat_tq = tq.dequantize(*tq.quantize(items))
    items_hat_mse = mse_q.dequantize(*mse_q.quantize(items))

    est_tq = (queries @ items_hat_tq.T).numpy()
    est_mse = (queries @ items_hat_mse.T).numpy()

    # Sample points
    n_sample = 50000
    idx = np.random.choice(true_ip.size, n_sample, replace=False)
    true_flat = true_ip.flatten()[idx]
    tq_flat = est_tq.flatten()[idx]
    mse_flat = est_mse.flatten()[idx]

    # --- Create contour figure ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    lim = 0.15
    grid_size = 200
    x_grid = np.linspace(-lim, lim, grid_size)
    y_grid = np.linspace(-lim, lim, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])

    # Left: MSE-only
    print("Computing MSE density...")
    # Use histogram2d for speed (KDE is slow on 50K points)
    H_mse, xedges, yedges = np.histogram2d(true_flat, mse_flat, bins=grid_size,
                                             range=[[-lim, lim], [-lim, lim]], density=True)
    # Gaussian smooth
    from scipy.ndimage import gaussian_filter
    H_mse_smooth = gaussian_filter(H_mse.T, sigma=3)

    levels = np.linspace(H_mse_smooth.max() * 0.05, H_mse_smooth.max() * 0.95, 8)

    cf1 = axes[0].contourf(X, Y, H_mse_smooth, levels=12, cmap='Greens', alpha=0.9)
    axes[0].contour(X, Y, H_mse_smooth, levels=8, colors='darkgreen', linewidths=0.5, alpha=0.6)
    # Reference lines
    axes[0].plot([-lim, lim], [-lim, lim], color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='y = x (no bias)')
    axes[0].plot([-lim, lim], [-lim*0.883, lim*0.883], color='black', linestyle='--', linewidth=2, label='α = 0.883 (shrinkage)')
    axes[0].set_xlabel('True Inner Product', fontsize=11)
    axes[0].set_ylabel('Estimated Inner Product', fontsize=11)
    axes[0].set_title('MSE-only (2-bit, d=512)\nBiased but tight', fontsize=12)
    axes[0].legend(fontsize=9, loc='upper left')
    axes[0].set_xlim(-lim, lim)
    axes[0].set_ylim(-lim, lim)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.2)

    # Right: TQ
    print("Computing TQ density...")
    H_tq, _, _ = np.histogram2d(true_flat, tq_flat, bins=grid_size,
                                  range=[[-lim, lim], [-lim, lim]], density=True)
    H_tq_smooth = gaussian_filter(H_tq.T, sigma=3)

    cf2 = axes[1].contourf(X, Y, H_tq_smooth, levels=12, cmap='Reds', alpha=0.9)
    axes[1].contour(X, Y, H_tq_smooth, levels=8, colors='darkred', linewidths=0.5, alpha=0.6)
    axes[1].plot([-lim, lim], [-lim, lim], color='black', linestyle='--', linewidth=2, label='y = x (unbiased)')
    axes[1].set_xlabel('True Inner Product', fontsize=11)
    axes[1].set_ylabel('Estimated Inner Product', fontsize=11)
    axes[1].set_title('TurboQuant (2-bit, d=512)\nUnbiased but wide', fontsize=12)
    axes[1].legend(fontsize=9, loc='upper left')
    axes[1].set_xlim(-lim, lim)
    axes[1].set_ylim(-lim, lim)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("/results/fig3_contour.png", dpi=200, bbox_inches="tight")
    plt.savefig("/results/fig3_contour.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig3_contour.png/pdf")

    volume.commit()


if __name__ == "__main__":
    with app.run():
        generate_contour_figure.remote()
