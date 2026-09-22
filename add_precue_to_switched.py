#!/usr/bin/env python3
"""
Attach 4s lead-in segments for switched stimuli.
- target_stretched/*.wav: prepend 4s padded precue and save to target/
- interference_stretched/*.wav: prepend 4s zero-padding and save to interference/
"""
import numpy as np
import csv
import os
import re
import soundfile as sf
from pathlib import Path
from glob import glob


def load_precue_from_name(name, target_length=4.0):
    """
    Load outputs/precue/{name}_cue.wav and pad/trim to target_length seconds.

    Returns (audio, sample_rate).
    """
    precue_path = os.path.join("outputs/precue", f"{name}_cue.wav")
    if not os.path.exists(precue_path):
        raise FileNotFoundError(f"Precue file not found: {precue_path}")
    
    audio, sr = sf.read(precue_path, dtype="float32")
    target_samples = int(target_length * sr)
    
    # Convert to mono.
    if audio.ndim > 1:
        audio = audio[:, 0]
    
    # Pad/trim to exactly target_length seconds.
    if len(audio) < target_samples:
        pad_width = target_samples - len(audio)
        audio = np.pad(audio, (0, pad_width), mode="constant")
    elif len(audio) > target_samples:
        # Trim if longer than target length.
        audio = audio[:target_samples]

    return audio, sr


def parse_filename(filename):
    """
    Parse day, block, stimulus_num from filename.
    Example: day1block1_stimulus16.wav -> (1, 1, 16)
    """
    pattern = r"day(\d+)block(\d+)_stimulus(\d+)\.wav"
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"Invalid filename format: {filename}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def load_name_from_csv(csv_path, day, block, stimulus_num):
    """Find speaker name by day, block, stimulus_num from CSV."""
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_day = row.get("day", "").strip()
            row_block = row.get("block", "").strip()
            row_stimulus_num = row.get("stimulus_num", "").strip()
            
            if (str(day) == row_day and 
                str(block) == row_block and 
                str(stimulus_num) == row_stimulus_num):
                name = row.get("name", "").strip()
                if name:
                    return name
    
    raise ValueError(f"Could not find name in CSV for day={day}, block={block}, stimulus_num={stimulus_num}.")


def process_target_files(
    input_dir="outputs/switched_stimulus/target_stretched",
    output_dir="outputs/switched_stimulus/target",
    csv_path="data/generated/switched/text_speaker_switched_target.csv",
    precue_length=4.0
):
    """Prepend precue to target_stretched files and save to target directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    wav_files = glob(os.path.join(input_dir, "*.wav"))
    total = len(wav_files)
    success = 0
    failed = 0
    
    print(f"[TARGET] Files to process: {total}")
    
    for wav_path in wav_files:
        try:
            filename = os.path.basename(wav_path)
            day, block, stimulus_num = parse_filename(filename)
            
            # Resolve speaker name from CSV.
            name = load_name_from_csv(csv_path, day, block, stimulus_num)
            
            # Load precue.
            precue, sr_precue = load_precue_from_name(name, target_length=precue_length)
            
            # Load target audio.
            target_audio, sr_target = sf.read(wav_path, dtype="float32")
            
            # Convert to mono.
            if target_audio.ndim > 1:
                target_audio = target_audio[:, 0]
            
            # Verify sample rates.
            if sr_precue != sr_target:
                raise ValueError(f"Sample-rate mismatch: precue={sr_precue}, target={sr_target}")
            
            # Prepend precue.
            target_with_precue = np.concatenate([precue, target_audio])
            
            # Build output path.
            output_path = os.path.join(output_dir, filename)
            
            # Save.
            sf.write(output_path, target_with_precue, sr_target)
            
            success += 1
            print(f"[OK] {filename} (name={name})")
            
        except Exception as e:
            failed += 1
            print(f"[FAIL] {filename}: {e}")
    
    print(f"\n[TARGET] Done: success={success}, failed={failed}, total={total}")


def process_interference_files(
    input_dir="outputs/switched_stimulus/interference_stretched",
    output_dir="outputs/switched_stimulus/interference",
    zero_padding_length=4.0
):
    """Prepend 4s zero padding to interference_stretched files."""
    os.makedirs(output_dir, exist_ok=True)
    
    wav_files = glob(os.path.join(input_dir, "*.wav"))
    total = len(wav_files)
    success = 0
    failed = 0
    
    print(f"[INTERFERENCE] Files to process: {total}")
    
    for wav_path in wav_files:
        try:
            filename = os.path.basename(wav_path)
            
            # Load interference audio.
            interference_audio, sr = sf.read(wav_path, dtype="float32")
            
            # Convert to mono.
            if interference_audio.ndim > 1:
                interference_audio = interference_audio[:, 0]
            
            # Build 4-second zero padding.
            padding_samples = int(zero_padding_length * sr)
            zero_padding = np.zeros(padding_samples, dtype=interference_audio.dtype)
            
            # Prepend zero padding.
            interference_with_padding = np.concatenate([zero_padding, interference_audio])
            
            # Build output path.
            output_path = os.path.join(output_dir, filename)
            
            # Save.
            sf.write(output_path, interference_with_padding, sr)
            
            success += 1
            print(f"[OK] {filename}")
            
        except Exception as e:
            failed += 1
            print(f"[FAIL] {filename}: {e}")
    
    print(f"\n[INTERFERENCE] Done: success={success}, failed={failed}, total={total}")


def main():
    """Process both target and interference switched files."""
    print("=" * 60)
    print("Start adding precue/zero-padding to switched_stimulus files")
    print("=" * 60)
    
    # Step 1: target files
    print("\n[Step 1] Processing target files...")
    process_target_files(
        input_dir="outputs/switched_stimulus/target_stretched",
        output_dir="outputs/switched_stimulus/target",
        csv_path="data/generated/switched/text_speaker_switched_target.csv",
        precue_length=4.0
    )
    
    # Step 2: interference files
    print("\n[Step 2] Processing interference files...")
    process_interference_files(
        input_dir="outputs/switched_stimulus/interference_stretched",
        output_dir="outputs/switched_stimulus/interference",
        zero_padding_length=4.0
    )
    
    print("\n" + "=" * 60)
    print("All processing completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
