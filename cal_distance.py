import os
import sys
import csv
import numpy as np
from glob import glob
from itertools import combinations
import matplotlib.pyplot as plt
import umap

# === Configuration ===
embedding_dir = "outputs/precue/embeddings"
pattern = os.path.join(embedding_dir, "*_cue.npy")
csv_path = "data/generated/pairwise_distances.csv"
top_k = 10

# === Load ===
npy_files = sorted(glob(pattern))
if not npy_files:
    raise FileNotFoundError(f"No files matched: {pattern}")

names = []
embs = []
for f in npy_files:
    arr = np.load(f)
    vec = np.ravel(arr)  # Ensure 1-D vector.
    if vec.ndim != 1:
        raise ValueError(f"Embedding in {f} is not 1-D after ravel: shape={arr.shape}")
    names.append(os.path.basename(f).replace("_cue.npy", ""))
    embs.append(vec)

embs = np.stack(embs)  # (N, 192)
N, dim = embs.shape
if dim == 0:
    raise ValueError("Empty embedding dimension.")

# === Compute all pair distances (L2 / dim) ===
pairs = []
for i, j in combinations(range(N), 2):
    d = np.linalg.norm(embs[i] - embs[j]) / dim
    pairs.append((names[i], names[j], float(d)))

# === Sort and save ===
pairs.sort(key=lambda x: x[2])  # Ascending order.

# Save CSV
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["name1", "name2", "distance"])
    writer.writerows(pairs)

# === Print: closest 10, furthest 10, median 10 ===
def print_pairs(title, items):
    print(f"\n[{title}]")
    for a, b, _ in items:
        print(f"{a} - {b}")

# Closest 10
closest = pairs[:top_k]

# Furthest 10 (displayed in descending order).
furthest = list(reversed(pairs[-top_k:]))

# 10 around the median
M = len(pairs)
mid = M // 2
window = top_k // 2
start = max(0, min(M - top_k, mid - window))
median_block = pairs[start:start + top_k]

print_pairs("Closest 10", closest)
print_pairs("Furthest 10", furthest)
print_pairs("Median 10", median_block)

print(f"\nTotal: {M}")
print(f"CSV: {csv_path}")
sys.exit()
reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, metric='euclidean', random_state=42)
embeddings_2d = reducer.fit_transform(embeddings)  # shape: (104, 2)

plt.figure(figsize=(12, 10))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=30, c='blue')

for i, label in enumerate(filenames):
    plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], label, fontsize=6)

plt.title("UMAP Projection of Speaker Embeddings with Filenames")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("embedding_umap.png")
plt.close()
