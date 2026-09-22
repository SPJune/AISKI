import csv
import re
from pathlib import Path

# Input/output paths
FILTERED_CSV = Path("data/generated/filtered_audio.csv")
TEXT_ALIGNED_CSV = Path("data/generated/text_aligned.csv")
OUTPUT_CSV = Path("data/generated/text_mod.csv")

# Keep output column order compatible with text_aligned.csv.
COLS = [
    "stimulus_num","name","main_topic","sub_topic","C","utterance","question",
    "option1","option2","option3","option4","answer","answer_index",
    "length_no_space","target_length"
]

# Extract num and name from stimulus{num}_{name}.wav.
pat = re.compile(r"^stimulus(\d+)_([\s\S]+)\.wav$", re.IGNORECASE)

# 1) Build (stimulus_num, name) set from filtered_audio.csv.
pairs = set()
with open(FILTERED_CSV, "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    # Assumes columns: filename, duration, stimulus_num.
    for row in reader:
        fname = (row.get("filename") or "").strip()
        stim_num = row.get("stimulus_num")
        # Parse from filename first.
        m = pat.match(fname)
        if m:
            num_from_name = int(m.group(1))
            name_from_name = m.group(2)
            pairs.add((num_from_name, name_from_name))
            continue
        # If filename pattern does not match, fallback to stimulus_num (name unknown -> skip).
        if stim_num is not None:
            try:
                stim_num_int = int(str(stim_num).strip())
            except ValueError:
                continue  # Not a number.
            # Name is required for matching, so skip here.
            # Add alternative filename rules here if needed.
            # print(f"Warning: skipped due to missing name parse: {fname}")
            continue

# 2) Filter matching rows from text_aligned.csv.
selected_rows = []
# Read with utf-8-sig to support possible BOM.
with open(TEXT_ALIGNED_CSV, "r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    # Normalize field handling for minor header issues.
    for row in reader:
        # Normalize values.
        try:
            stim = int(str(row.get("stimulus_num", "")).strip())
        except ValueError:
            continue
        name = (row.get("name") or "").strip()
        if (stim, name) in pairs:
            # Keep only output columns (missing keys become empty string).
            selected_rows.append({k: row.get(k, "") for k in COLS})

# 3) Save as text_mod.csv.
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=COLS)
    writer.writeheader()
    writer.writerows(selected_rows)

print(f"Extraction complete: {OUTPUT_CSV} (rows: {len(selected_rows)})")
