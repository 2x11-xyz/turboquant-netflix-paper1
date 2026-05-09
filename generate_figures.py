"""
Generate publication-ready figures for the paper.
Figure 1: Recall@10 gap vs dimension (line plot, one line per bit-width)
Figure 2: Score scatter (true vs estimated) for MSE-only vs TQ

Run: `modal run generate_figures.py`
"""
import modal

app = modal.App("tq-figures")
volume = modal.Volume.from_name("turboquant-netflix-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scipy", "matplotlib")
    .add_local_file("turboquant_impl.py", "/root/turboquant_impl.py")
)


@app.function(image=image, volumes={"/results": volume}, timeout=600)
def generate_all_figures():
    import json
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    # Load results
    with open("/results/dim_sweep_results.json") as f:
        sweep_23 = json.load(f)
    with open("/results/dim_sweep_4bit_results.json") as f:
        sweep_48 = json.load(f)
    with open("/results/glove_normalized_results.json") as f:
        glove = json.load(f)
    with open("/results/highd_results.json") as f:
        highd = json.load(f)

    # =====================================================================
    # FIGURE 1: Recall@10 gap (MSE - TQ) vs dimension, one line per bit-width
    # =====================================================================
    dims = [50, 128, 256, 512, 1024, 2048]
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    colors = {2: '#d62728', 3: '#ff7f0e', 4: '#2ca02c', 8: '#1f77b4'}
    markers = {2: 'o', 3: 's', 4: '^', 8: 'D'}

    for bits, data_src in [(2, sweep_23), (3, sweep_23), (4, sweep_48), (8, sweep_48)]:
        gaps = []
        stds = []
        for d in dims:
            r = data_src[str(d)][str(bits)]
            gap = r['recall10_mse']['mean'] - r['recall10_tq']['mean']
            # Propagate std: std of difference ~ sqrt(std_mse^2 + std_tq^2)
            std = np.sqrt(r['recall10_mse']['std']**2 + r['recall10_tq']['std']**2)
            gaps.append(gap)
            stds.append(std)

        ax.errorbar(dims, gaps, yerr=stds, marker=markers[bits], color=colors[bits],
                    label=f'{bits}-bit', linewidth=2, markersize=7, capsize=3)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Embedding Dimension (d)', fontsize=12)
    ax.set_ylabel('Recall@10 Gap (MSE-only minus TQ)', fontsize=12)
    ax.set_xscale('log', base=2)
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.legend(fontsize=11, loc='center right')
    ax.set_ylim(-0.02, 0.35)
    ax.grid(True, alpha=0.3)
    ax.set_title('MSE-Optimal Quantization Dominates at All Dimensions and Bit-Widths', fontsize=11)
    
    plt.tight_layout()
    plt.savefig("/results/fig1_recall_gap.png", dpi=200, bbox_inches="tight")
    plt.savefig("/results/fig1_recall_gap.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig1_recall_gap.png/pdf")

    # =====================================================================
    # FIGURE 2: Recall@10 (absolute) for MSE vs TQ across bit-widths
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: d=512 (representative)
    d_show = 512
    bits_all = [2, 3, 4, 8]
    
    tq_recalls = []
    mse_recalls = []
    for bits in bits_all:
        src = sweep_23 if bits in [2, 3] else sweep_48
        tq_recalls.append(src[str(d_show)][str(bits)]['recall10_tq']['mean'])
        mse_recalls.append(src[str(d_show)][str(bits)]['recall10_mse']['mean'])

    x = np.arange(len(bits_all))
    w = 0.35
    axes[0].bar(x - w/2, mse_recalls, w, label='MSE-only', color='#2ca02c', alpha=0.8)
    axes[0].bar(x + w/2, tq_recalls, w, label='TurboQuant', color='#d62728', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'{b}-bit' for b in bits_all])
    axes[0].set_ylabel('Recall@10', fontsize=12)
    axes[0].set_xlabel('Bit-width', fontsize=12)
    axes[0].set_title(f'Recall@10 at d={d_show} (Gaussian)', fontsize=11)
    axes[0].legend(fontsize=11)
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Right: Alpha vs predicted from D_mse
    predicted_alpha = {2: 1 - 0.117, 3: 1 - 0.034, 4: 1 - 0.009, 8: 1 - 0.0001}
    measured_alpha = {}
    for bits in bits_all:
        src = sweep_23 if bits in [2, 3] else sweep_48
        # Average alpha across all dimensions
        alphas = [src[str(d)][str(bits)]['alpha']['mean'] for d in dims]
        measured_alpha[bits] = np.mean(alphas)

    pred_vals = [predicted_alpha[b] for b in bits_all]
    meas_vals = [measured_alpha[b] for b in bits_all]
    
    axes[1].scatter(pred_vals, meas_vals, s=100, zorder=5, c=[colors[b] for b in bits_all])
    for i, b in enumerate(bits_all):
        axes[1].annotate(f'  {b}-bit', (pred_vals[i], meas_vals[i]), fontsize=10)
    
    lims = [0.87, 1.005]
    axes[1].plot(lims, lims, 'k--', alpha=0.5, label='y = x (perfect match)')
    axes[1].set_xlim(lims)
    axes[1].set_ylim(lims)
    axes[1].set_xlabel('Predicted α = 1 − D_mse (from TQ Theorem 1)', fontsize=11)
    axes[1].set_ylabel('Measured α (empirical)', fontsize=11)
    axes[1].set_title('Shrinkage Factor Matches TQ Theory', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')

    plt.tight_layout()
    plt.savefig("/results/fig2_bars_alpha.png", dpi=200, bbox_inches="tight")
    plt.savefig("/results/fig2_bars_alpha.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig2_bars_alpha.png/pdf")

    # =====================================================================
    # FIGURE 3: Score scatter (true vs estimated) MSE vs TQ
    # =====================================================================
    import sys
    sys.path.insert(0, "/root")
    import torch
    from turboquant_impl import TurboQuantIP, MSEOnlyQuantizer

    # Generate fresh d=512, 2-bit data for scatter
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

    # Single seed (deployment reality)
    seed = 0
    tq = TurboQuantIP(dim=d, bits=bits, seed=seed)
    mse_q = MSEOnlyQuantizer(dim=d, bits=bits, seed=seed)

    items_hat_tq = tq.dequantize(*tq.quantize(items))
    items_hat_mse = mse_q.dequantize(*mse_q.quantize(items))

    est_tq = (queries @ items_hat_tq.T).numpy()
    est_mse = (queries @ items_hat_mse.T).numpy()

    # Sample points for scatter
    n_sample = 20000
    idx = np.random.choice(true_ip.size, n_sample, replace=False)
    true_flat = true_ip.flatten()[idx]
    tq_flat = est_tq.flatten()[idx]
    mse_flat = est_mse.flatten()[idx]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: MSE-only
    axes[0].scatter(true_flat, mse_flat, s=1, alpha=0.3, c='#2ca02c', rasterized=True)
    axes[0].plot([-0.15, 0.15], [-0.15*0.883, 0.15*0.883], 'k--', alpha=0.7, 
                 label=f'α={0.883:.3f} (predicted)')
    axes[0].plot([-0.15, 0.15], [-0.15, 0.15], 'gray', alpha=0.3, linestyle=':')
    axes[0].set_xlabel('True Inner Product', fontsize=11)
    axes[0].set_ylabel('Estimated Inner Product', fontsize=11)
    axes[0].set_title('MSE-only (2-bit, d=512)\nTight spread, predictable shrinkage', fontsize=11)
    axes[0].legend(fontsize=10, loc='upper left')
    axes[0].set_xlim(-0.15, 0.15)
    axes[0].set_ylim(-0.15, 0.15)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.2)

    # Right: TQ
    axes[1].scatter(true_flat, tq_flat, s=1, alpha=0.3, c='#d62728', rasterized=True)
    axes[1].plot([-0.15, 0.15], [-0.15, 0.15], 'k--', alpha=0.7, label='y = x (unbiased)')
    axes[1].set_xlabel('True Inner Product', fontsize=11)
    axes[1].set_ylabel('Estimated Inner Product', fontsize=11)
    axes[1].set_title('TurboQuant (2-bit, d=512)\nCentered on truth, but wide spread', fontsize=11)
    axes[1].legend(fontsize=10, loc='upper left')
    axes[1].set_xlim(-0.15, 0.15)
    axes[1].set_ylim(-0.15, 0.15)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("/results/fig3_scatter.png", dpi=200, bbox_inches="tight")
    plt.savefig("/results/fig3_scatter.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig3_scatter.png/pdf")

    volume.commit()
    print("\nAll figures generated!")


if __name__ == "__main__":
    with app.run():
        generate_all_figures.remote()
