"""
Minimal script to download MovieLens-1M to Modal Volume (persistent storage).
Run once: `modal run movie_lens_download.py`
"""
import modal
import torch
import pandas as pd
import requests
import zipfile
import os

# Modal setup
app = modal.App("movielens-download")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "pandas", "requests")
)
# Use same volume as experiment.py for compatibility
volume = modal.Volume.from_name("turboquant-netflix-results", create_if_missing=True)

@app.function(image=image, volumes={"/results": volume}, timeout=3600)
def download_and_save():
    """Download MovieLens-1M, process, save to Modal Volume"""
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = "/tmp/ml-1m.zip"
    extract_path = "/tmp/ml-1m"
    
    # Download
    print("Downloading MovieLens-1M...")
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
    
    # Create user-item matrix
    user_ids = df["user_id"].unique()
    item_ids = df["item_id"].unique()
    n_users = len(user_ids)
    n_items = len(item_ids)
    
    user_map = {id: idx for idx, id in enumerate(user_ids)}
    item_map = {id: idx for idx, id in enumerate(item_ids)}
    
    X = torch.zeros((n_users, n_items))
    for _, row in df.iterrows():
        u = user_map[row["user_id"]]
        i = item_map[row["item_id"]]
        X[u, i] = 1.0
    
    # Save to mounted volume (persistent storage)
    torch.save({
        "X": X,
        "user_map": user_map,
        "item_map": item_map
    }, "/results/movielens_1m.pt")
    
    # Commit volume to persist changes
    volume.commit()
    return f"Saved MovieLens-1M ({n_users} users, {n_items} items) to Modal Volume"

if __name__ == "__main__":
    with app.run():
        print(download_and_save.remote())
