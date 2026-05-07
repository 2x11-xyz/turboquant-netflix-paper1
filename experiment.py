import modal
import torch
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from turboquant import TurboQuantProd  # From PyPI: turboquant-pro
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import recall_score, ndcg_score
import requests
import zipfile
import os

# Modal setup (modern API)
app = modal.App("turboquant-netflix-experiments")
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "turboquant-pro",
        "requests"
    )
)
volume = modal.Volume.from_name("turboquant-netflix-results", create_if_missing=True)

# Configuration
K = 50  # Latent dimensions
LAMBDA = 0.01  # Regularization strength
BITS = [2, 3]  # TurboQuant bit-widths (integer for library compatibility)
T_VALS = [0, 0.5, 1, 2, 5]  # D scaling skew levels
M_SEEDS = 10  # TurboQuant random seeds for variance estimation
N_PAIRS = 10  # Number of user-item pairs to sample for variance estimation


def download_movielens() -> str:
    """Download MovieLens-1M and save to volume"""
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = "/tmp/ml-1m.zip"
    extract_path = "/tmp/ml-1m"
    
    # Download with error handling
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(zip_path, "wb") as f:
        f.write(response.content)
    
    # Extract
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    
    # Load ratings
    ratings_path = os.path.join(extract_path, "ml-1m", "ratings.dat")
    df = pd.read_csv(
        ratings_path,
        sep="::",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python"
    )
    
    # Create user-item matrix (n_users × n_items)
    user_ids = df["user_id"].unique()
    item_ids = df["item_id"].unique()
    n_users = len(user_ids)
    n_items = len(item_ids)
    
    # Map IDs to contiguous indices
    user_map = {id: idx for idx, id in enumerate(user_ids)}
    item_map = {id: idx for idx, id in enumerate(item_ids)}
    
    # Fill matrix (1 for interacted items, 0 otherwise for implicit feedback)
    X = torch.zeros((n_users, n_items))
    for _, row in df.iterrows():
        u = user_map[row["user_id"]]
        i = item_map[row["item_id"]]
        X[u, i] = 1.0
    
    # Save to volume
    torch.save({
        "X": X,
        "user_map": user_map,
        "item_map": item_map
    }, "/results/movielens_1m.pt")
    volume.commit()
    return f"Downloaded MovieLens-1M: {n_users} users, {n_items} items"


def train_mf(X: torch.Tensor, reg_scheme: str = "eq1") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Train linear MF model with Netflix's regularization schemes.
    Returns: (A, B) where user embeddings = X @ A, item embeddings = B
    """
    n, p = X.shape
    k = K
    
    # Initialize with requires_grad=True (fixed bug)
    A = torch.nn.Parameter(torch.randn(p, k) * 0.01)
    B = torch.nn.Parameter(torch.randn(p, k) * 0.01)
    optimizer = torch.optim.Adam([A, B], lr=0.001)
    
    # Training loop
    for epoch in range(100):
        optimizer.zero_grad()
        
        # Reconstruction loss
        recon = X @ A @ B.T  # X @ (A B^T)
        loss = ((X - recon) ** 2).sum()
        
        # Regularization (Eq.1 is correct as written; Eq.2 is correct)
        if reg_scheme == "eq1":
            # Netflix Eq.1: penalty on ||AB^T||_F^2
            loss += LAMBDA * ((A @ B.T) ** 2).sum()
        elif reg_scheme == "eq2":
            # Netflix Eq.2: penalty on ||XA||_F^2 + ||B||_F^2
            loss += LAMBDA * ((X @ A) ** 2).sum() + LAMBDA * (B ** 2).sum()
        else:
            raise ValueError(f"Unknown reg_scheme: {reg_scheme}")
        
        loss.backward()
        optimizer.step()
    
    # Detach to avoid gradient issues
    return A.detach(), B.detach()


def generate_D(t: float, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate diagonal scaling matrix D_t with skew t, using fixed z"""
    diags = torch.exp(t * z)
    D = torch.diag(diags)
    return D, diags


@app.function(image=image, volumes={"/results": volume}, timeout=3600)
def run_experiments():
    """Main experiment loop: replicate Netflix + TurboQuant"""
    # Load data
    data = torch.load("/results/movielens_1m.pt")
    X = data["X"]  # User-item matrix
    n_users, n_items = X.shape
    
    results = []
    
    # Generate fixed z for all t values (fixed bug: same z per t)
    z = torch.randn(K)
    
    for reg_scheme in ["eq1", "eq2"]:
        # Train MF model
        A, B = train_mf(X, reg_scheme)
        user_emb = X @ A  # n_users × k
        item_emb = B  # n_items × k
        
        for t in T_VALS:
            D, diags = generate_D(t, z)  # Use fixed z
            D_inv = torch.diag(1.0 / diags)
            
            # Scale embeddings (Netflix Eq.3)
            user_emb_scaled = user_emb @ D  # u^(D) = uD
            item_emb_scaled = item_emb @ D_inv  # v^(D) = vD^{-1}
            
            # Sample user-item pairs (without replacement)
            np.random.seed(42)
            user_idxs = np.random.choice(n_users, N_PAIRS, replace=False)
            item_idxs = np.random.choice(n_items, N_PAIRS, replace=False)
            
            # 1. Netflix cosine similarity (arbitrary)
            cos_sim = []
            for u in user_idxs:
                for v in item_idxs:
                    u_vec = user_emb_scaled[u]
                    v_vec = item_emb_scaled[v]
                    denom = torch.norm(u_vec) * torch.norm(v_vec) + 1e-8
                    cos = torch.dot(u_vec, v_vec) / denom
                    cos_sim.append(cos.item())
            
            # 2. Netflix raw dot product (invariant)
            raw_dot = []
            for u in user_idxs:
                for v in item_idxs:
                    dot = torch.dot(user_emb_scaled[u], item_emb_scaled[v])
                    raw_dot.append(dot.item())
            
            # 3. TurboQuant (one-sided, quantize items only)
            # Sample smaller subset for variance estimation
            pair_keys = [(u, v) for u in user_idxs[:5] for v in item_idxs[:5]]
            tq_pair_dots = {b: {k: [] for k in pair_keys} for b in BITS}
            
            for b in BITS:
                # Initialize TurboQuant for item embeddings (dims = k)
                tq = TurboQuantProd(d=K, bits=b)
                
                for seed in range(M_SEEDS):
                    # Quantize item embeddings
                    item_quant = tq.quantize(item_emb_scaled)
                    # Dequantize to get unbiased estimator (fixed: add dequantize)
                    item_dequant = tq.dequantize(item_quant)
                    
                    # Compute dot products per pair
                    for u, v in pair_keys:
                        u_vec = user_emb_scaled[u]  # Full-precision user
                        v_dequant = item_dequant[v]  # Dequantized item
                        dot = torch.dot(u_vec, v_dequant).item()
                        tq_pair_dots[b][(u, v)].append(dot)
                
                # Compute variance per pair, then average (fixed variance bug)
                pair_vars = [np.var(dots) for dots in tq_pair_dots[b].values()]
                avg_var = np.mean(pair_vars) if pair_vars else 0.0
            
            # Save results
            results.append({
                "reg_scheme": reg_scheme,
                "t": t,
                "kappa_D": (torch.max(diags) / torch.min(diags)).item(),  # Fixed: .item()
                "cos_sim_mean": np.mean(cos_sim),
                "cos_sim_var": np.var(cos_sim),
                "raw_dot_mean": np.mean(raw_dot),
                "raw_dot_var": np.var(raw_dot),
                "tq_avg_var": avg_var,
                "tq_pair_dots": tq_pair_dots
            })
    
    # Save results to volume
    torch.save(results, "/results/experiment_results.pt")
    volume.commit()
    
    # Generate heatmaps
    generate_heatmaps(results)
    volume.commit()
    
    return "Experiments complete"


def generate_heatmaps(results):
    """Generate heatmaps replicating Netflix + TurboQuant"""
    # Reorder results: [eq1_t0, eq1_t0.5..., eq2_t0, eq2_t0.5...]
    # Reshape to (2, 5) then transpose to (5, 2) for correct axis (fixed reshape bug)
    cos_data = np.array([r["cos_sim_mean"] for r in results]).reshape(2, len(T_VALS)).T
    raw_data = np.array([r["raw_dot_mean"] for r in results]).reshape(2, len(T_VALS)).T
    tq_var_data = np.array([r["tq_avg_var"] for r in results]).reshape(2, len(T_VALS)).T
    
    # Plot 3 heatmaps (Netflix 2 + TurboQuant 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.heatmap(cos_data, ax=axes[0], xticklabels=["eq1", "eq2"], yticklabels=T_VALS,
                cmap="viridis", cbar_kws={"label": "Cosine Similarity"})
    axes[0].set_title("Netflix: Cosine Similarity (Arbitrary)")
    axes[0].set_xlabel("Regularization")
    axes[0].set_ylabel("D Skew (t)")
    
    sns.heatmap(raw_data, ax=axes[1], xticklabels=["eq1", "eq2"], yticklabels=T_VALS,
                cmap="viridis", cbar_kws={"label": "Raw Dot Product"})
    axes[1].set_title("Netflix: Raw Dot Product (Stable)")
    axes[1].set_xlabel("Regularization")
    axes[1].set_ylabel("D Skew (t)")
    
    sns.heatmap(tq_var_data, ax=axes[2], xticklabels=["eq1", "eq2"], yticklabels=T_VALS,
                cmap="viridis", cbar_kws={"label": "TurboQuant Variance"})
    axes[2].set_title("TurboQuant: Dot Product Variance (Scales with D)")
    axes[2].set_xlabel("Regularization")
    axes[2].set_ylabel("D Skew (t)")
    
    plt.tight_layout()
    plt.savefig("/results/netflix_turboquant_heatmaps.png", dpi=300)
    plt.close()
    
    return "Heatmaps saved to /results"


if __name__ == "__main__":
    with app.run():
        # Step 1: Download data
        print(download_movielens())
        # Step 2: Run experiments
        print(run_experiments())
