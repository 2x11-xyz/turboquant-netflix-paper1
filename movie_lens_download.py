"""
Minimal script to download MovieLens-1M to Modal Volume (persistent storage).
Run once: `modal run movie_lens_download.py`
"""
import modal
import requests
import zipfile
import os

app = modal.App("movielens-download")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "pandas", "requests")
)
volume = modal.Volume.from_name("turboquant-netflix-results", create_if_missing=True)


@app.function(image=image, volumes={"/results": volume}, timeout=3600)
def download_and_save():
    """Download MovieLens-1M, process, save to Modal Volume"""
    import torch
    import pandas as pd

    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = "/tmp/ml-1m.zip"
    extract_path = "/tmp/ml-1m"

    # Download in chunks (avoids loading full zip into RAM)
    print("Downloading MovieLens-1M...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Extract
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    # Load ratings
    ratings_path = os.path.join(extract_path, "ml-1m", "ratings.dat")
    df = pd.read_csv(
        ratings_path,
        sep="::",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )

    # Build contiguous ID maps
    user_ids = df["user_id"].unique()
    item_ids = df["item_id"].unique()
    n_users = len(user_ids)
    n_items = len(item_ids)
    user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_map = {iid: idx for idx, iid in enumerate(item_ids)}

    # Build user-item matrix via vectorized assignment (fixes iterrows slowness)
    u_indices = df["user_id"].map(user_map).values.copy()
    i_indices = df["item_id"].map(item_map).values.copy()
    X = torch.zeros((n_users, n_items))
    X[u_indices, i_indices] = 1.0

    # Save to volume and commit
    torch.save({"X": X, "user_map": user_map, "item_map": item_map},
               "/results/movielens_1m.pt")
    volume.commit()
    print(f"Saved: {n_users} users, {n_items} items → /results/movielens_1m.pt")
    return f"Done: {n_users} users, {n_items} items"


if __name__ == "__main__":
    with app.run():
        print(download_and_save.remote())
