import pandas as pd
import random

def sample_pairs(file, required_speakers, n_samples=200, seed=42):
    random.seed(seed)
    
    # Load input data.
    df = pd.read_csv(file)
    
    # Filter by distance > 2.
    df = df[df['distance'] > 2]
    
    # Collect unique speakers.
    speakers = set(df['name1']).union(set(df['name2']))
    
    if len(speakers) < required_speakers:
        raise ValueError(f"{file} does not satisfy required speakers ({required_speakers}). Current: {len(speakers)}")
    
    # Select samples to cover all speakers first.
    selected = []
    covered = set()
    remaining = df.copy()
    
    # Secure at least one pair per speaker.
    for spk in speakers:
        candidates = remaining[(remaining['name1'] == spk) | (remaining['name2'] == spk)]
        if not candidates.empty:
            row = candidates.sample(1, random_state=seed)
            selected.append(row)
            covered.update([row['name1'].values[0], row['name2'].values[0]])
            remaining = remaining.drop(row.index)
    
    # Fill remaining rows until n_samples.
    needed = n_samples - len(pd.concat(selected)) if selected else n_samples
    if needed > 0:
        extra = remaining.sample(needed, random_state=seed)
        selected.append(extra)
    
    return pd.concat(selected)

# Process each gender group.
mm = sample_pairs("data/generated/MM_distance.csv", required_speakers=93, n_samples=200, seed=42)
ff = sample_pairs("data/generated/FF_distance.csv", required_speakers=77, n_samples=200, seed=43)
mf = sample_pairs("data/generated/MF_distance.csv", required_speakers=170, n_samples=200, seed=44)

# Merge and save.
pairs = pd.concat([mm, ff, mf])
pairs.to_csv("data/generated/pairs.csv", index=False)

print("Saved: data/generated/pairs.csv")
