import os
import re
import pandas as pd
import wave

def clean_name(name: str) -> str:
    """Keep only alphabetic characters in a name."""
    return "".join(re.findall(r"[A-Za-z]", name))

def get_wav_duration(path):
    """Return WAV duration in seconds."""
    with wave.open(path, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / float(rate)
    return duration

# Input/output paths
input_csv = "data/reference/typecast_voice_info.csv"
output_csv = "data/generated/voice_info.csv"
wav_dir = "outputs/precue"
target_sec = 30

# Load CSV
df = pd.read_csv(input_csv)

rows = []
Cs = []
for _, row in df.iterrows():
    orig_name = row["name"]
    voice_id = row["voice_id"]
    gender = row["gender"]

    # Cleaned name used by WAV filenames.
    clean = clean_name(orig_name)
    wav_path = os.path.join(wav_dir, f"{clean}_cue.wav")

    if os.path.exists(wav_path):
        duration = get_wav_duration(wav_path)
        velocity = round(duration / 11, 4)  # Rounded to 4 decimals.
        C = round(target_sec/velocity)
    else:
        velocity = None
        C = None

    rows.append({
        "name": clean,  # Final name follows WAV filename convention.
        "voice_id": voice_id,
        "gender": gender,
        "velocity": velocity,
        "C": C
    })

# Build DataFrame and save
out_df = pd.DataFrame(rows)
out_df.to_csv(output_csv, index=False)

print(f"Saved: {output_csv}")
