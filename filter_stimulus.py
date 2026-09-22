import os
import wave
import csv
import re

# Duration filter thresholds (seconds)
MIN_LEN = 25
MAX_LEN = 35

pattern = re.compile(r"stimulus(\d+)_")  # Regex to extract stimulus_num.
results = []

for fname in os.listdir("outputs/stimulus"):
    if fname.lower().endswith('.wav'):
        try:
            with wave.open(os.path.join("outputs/stimulus", fname), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)

                if duration < MIN_LEN or duration > MAX_LEN:
                    print(f"{fname} - {duration:.2f}s")

                    # Extract stimulus_num.
                    match = pattern.search(fname)
                    if match:
                        stim_num = int(match.group(1))
                    else:
                        stim_num = 999999  # Push unmatched files to the end.

                    results.append((fname, duration, stim_num))
        except wave.Error as e:
            print(f"{fname} - error ({e})")

# Sort by stimulus_num.
results.sort(key=lambda x: x[2])

# Save CSV.
if results:
    with open("data/generated/filtered_audio.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "duration", "stimulus_num"])
        for fname, duration, stim_num in results:
            writer.writerow([fname, duration, stim_num])
    print("\nSaved data/generated/filtered_audio.csv sorted by stimulus_num.")
else:
    print("\nNo files matched the filter criteria.")
