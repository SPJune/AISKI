import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import numpy as np

# Parameters
INPUT_CSV = 'data/generated/topics.csv'
OUTPUT_JSON = 'data/generated/grouped_topics.json'
SIMILARITY_THRESHOLD = 0.82  # Tune as needed.

# 1) Load data
df = pd.read_csv(INPUT_CSV)
main_topics = df['main_topic'].dropna().unique().tolist()

# 2) Encode topics
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
embeddings = model.encode(main_topics, convert_to_tensor=False)

# 3) Compute similarity matrix
similarity_matrix = cosine_similarity(embeddings)

# 4) Group similar topics using Union-Find
parent = list(range(len(main_topics)))

def find(u):
    while parent[u] != u:
        parent[u] = parent[parent[u]]
        u = parent[u]
    return u

def union(u, v):
    pu, pv = find(u), find(v)
    if pu != pv:
        parent[pu] = pv

for i in range(len(main_topics)):
    for j in range(i + 1, len(main_topics)):
        if similarity_matrix[i][j] >= SIMILARITY_THRESHOLD:
            union(i, j)

# 5) Build grouped result
groups = defaultdict(list)
for idx, topic in enumerate(main_topics):
    root = find(idx)
    groups[root].append(topic)

# 6) Pick representative and sort
grouped_topics = {}
for group in groups.values():
    if len(group) <= 1:
        continue
    representative = min(group, key=len)
    grouped_topics[representative] = sorted(group)

# 7) Save JSON
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(grouped_topics, f, ensure_ascii=False, indent=2)

print(f"Saved {len(grouped_topics)} similar topic groups to '{OUTPUT_JSON}'.")
