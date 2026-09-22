import os
import glob
import re
import csv
import argparse
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import parselmouth
from parselmouth.praat import call

def _parselmouth_stretch(audio_mono: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    sound = parselmouth.Sound(audio_mono, sr)
    manip = call(sound, "To Manipulation...", 0.01, 50, 600)
    dt = call("Create DurationTier", "dur", sound.xmin, sound.xmax)
    call(dt, "Add point", sound.xmin, ratio)
    call(dt, "Add point", sound.xmax, ratio)
    call([dt, manip], "Replace duration tier")
    stretched = call(manip, "Get resynthesis (overlap-add)")
    return np.asarray(stretched.values).squeeze()

def stretch_with_cap_and_pad(audio: np.ndarray, sr: int, target_sec: float, max_ratio: float = 1.2):
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    audio = np.ascontiguousarray(audio.astype(np.float32))

    current_sec = len(audio) / sr
    if current_sec <= 0:
        raise ValueError("Input audio length is 0 seconds.")
    desired_ratio = target_sec / current_sec

    applied_ratio = float(min(max(desired_ratio, 1e-6), max_ratio))

    stretched = _parselmouth_stretch(audio, sr, applied_ratio)
    stretched_sec = len(stretched) / sr

    if stretched_sec < target_sec:
        total_pad_sec = target_sec - stretched_sec
        half = total_pad_sec / 2.0
        front_pad_sec = min(half, 1.0)
        back_pad_sec = total_pad_sec - front_pad_sec
        front_pad = int(round(front_pad_sec * sr))
        back_pad = int(round(back_pad_sec * sr))
        out = np.pad(stretched, (front_pad, back_pad), mode="constant", constant_values=0.0)
    else:
        target_len = int(round(target_sec * sr))
        out = stretched[:target_len]

    return out.astype(np.float32), applied_ratio

def parse_stimulus_num(fname: str):
    """Extract stimulus number from supported filename formats."""
    m = re.match(r"stimulus(\d+)_(.+)\.wav", fname)
    if m:
        return m.group(1), m.group(2)
    m = re.match(r"day\d+block\d+_stimulus(\d+)\.wav", fname)
    if m:
        return m.group(1), ""
    return None, None


def main(input_dir="pilot", output_dir="pilot_stretched", target_sec=30.0, info_csv="pilot_stretch_info.csv"):
    os.makedirs(output_dir, exist_ok=True)
    wav_files = glob.glob(os.path.join(input_dir, "*.wav"))
    if not wav_files:
        print(f"No .wav files found in {input_dir}")
        return

    info_dict = {}  # {stimulus_num: {name: ratio}}

    for wav_path in wav_files:
        try:
            fname = os.path.basename(wav_path)
            # Extract stimulus_num and optional speaker name.
            stimulus_num, name = parse_stimulus_num(fname)
            if stimulus_num is None:
                print(f"Filename not matched: {fname}")
                continue

            audio, sr = librosa.load(wav_path, sr=None, mono=False)
            out_audio, applied_ratio = stretch_with_cap_and_pad(audio, sr, target_sec, max_ratio=1.2)

            # Final length correction.
            target_len = int(round(target_sec * sr))
            if len(out_audio) < target_len:
                out_audio = np.pad(out_audio, (0, target_len - len(out_audio)), mode="constant", constant_values=0.0)
            elif len(out_audio) > target_len:
                out_audio = out_audio[:target_len]

            out_path = os.path.join(output_dir, fname)
            sf.write(out_path, out_audio, sr, subtype="PCM_16")
            print(f"Saved: {out_path} | ratio={applied_ratio:.3f}")

            # Save ratio info.
            if name:
                if stimulus_num not in info_dict:
                    info_dict[stimulus_num] = {}
                info_dict[stimulus_num][name] = applied_ratio
        except Exception as e:
            print(f"[ERROR] {wav_path}: {e}")

    # Save CSV.
    if info_dict:
        with open(info_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["stimulus_num", "name1", "ratio1", "name2", "ratio2"])
            for stimulus_num, subdict in sorted(info_dict.items(), key=lambda x: int(x[0])):
                if len(subdict) == 2:
                    names = sorted(subdict.keys())
                    writer.writerow([
                        stimulus_num,
                        names[0], f"{subdict[names[0]]:.4f}",
                        names[1], f"{subdict[names[1]]:.4f}"
                    ])
                else:
                    # If only one name exists, keep row and fill blanks.
                    names = list(subdict.keys())
                    writer.writerow([
                        stimulus_num,
                        names[0], f"{subdict[names[0]]:.4f}",
                        "", ""
                    ])
        print(f"Stretch info saved to {info_csv}")


def process_stimulus_eng_root(root_dir: str, target_sec: float = 30.0):
    """
    Process outputs/stimulus_eng/day{i}block{j}/target_raw/*.wav
    -> outputs/stimulus_eng/day{i}block{j}/target_stretched/*.wav
    """
    root = Path(root_dir)
    target_raw_dirs = sorted(root.glob("day*block*/target_raw"))
    if not target_raw_dirs:
        print(f"No target_raw directories found under {root}")
        return

    for raw_dir in target_raw_dirs:
        block_dir = raw_dir.parent
        stretched_dir = block_dir / "target_stretched"
        info_csv = Path("data/generated") / f"{block_dir.name}_stretch_info.csv"
        print(f"\n=== Processing {raw_dir} -> {stretched_dir} ===")
        main(
            input_dir=str(raw_dir),
            output_dir=str(stretched_dir),
            target_sec=target_sec,
            info_csv=str(info_csv),
        )


def _read_mono(path: Path):
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr


def _resample_if_needed(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    return librosa.resample(audio, orig_sr=src_sr, target_sr=dst_sr).astype(np.float32)


def _fit_to_seconds(audio: np.ndarray, sr: int, sec: float) -> np.ndarray:
    target_len = int(round(sec * sr))
    if len(audio) < target_len:
        return np.pad(audio, (0, target_len - len(audio)), mode="constant", constant_values=0.0).astype(np.float32)
    return audio[:target_len].astype(np.float32)


def _interference_num(num: int) -> int:
    if 1 <= num <= 15:
        return num + 15
    if 16 <= num <= 30:
        return num - 15
    raise ValueError(f"stimulus number out of supported range [1, 30]: {num}")


def _load_name_target_map(csv_path: Path):
    mapping = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stimulus_num = row.get("stimulus_num")
            name_target = row.get("name_target")
            if not stimulus_num or not name_target:
                continue
            mapping[int(stimulus_num)] = name_target
    return mapping


def process_stimulus_eng_final_root(
    stim_root: str = "outputs/stimulus_eng",
    precue_dir: str = "outputs/precue_eng",
    output_root: str = "outputs/stimulus_eng_final",
    reference_dir: str = "data/reference_eng",
    precue_sec: float = 4.0,
):
    """
    Build target/interference/mixture:
    - input target stretched: outputs/stimulus_eng/day{i}block{j}/target_stretched/day{i}block{j}_stimulus{num:02d}.wav
    - input precue: outputs/precue_eng/{name_target}_cue.wav (name_target from data/reference_eng/day{i}block{j}.csv)
    - output:
      outputs/stimulus_eng_final/day{i}block{j}/target/day{i}block{j}_stimulus{num:02d}.wav
      outputs/stimulus_eng_final/day{i}block{j}/interference/day{i}block{j}_stimulus{num:02d}.wav
      outputs/stimulus_eng_final/day{i}block{j}/mixture/day{i}block{j}_stimulus{num:02d}.wav
    """
    stim_root_path = Path(stim_root)
    precue_dir_path = Path(precue_dir)
    output_root_path = Path(output_root)
    reference_dir_path = Path(reference_dir)

    block_dirs = sorted(stim_root_path.glob("day*block*"))
    if not block_dirs:
        print(f"No day*block* directories found under {stim_root_path}")
        return

    total_done = 0
    total_skipped = 0

    for block_dir in block_dirs:
        block_name = block_dir.name
        target_stretched_dir = block_dir / "target_stretched"
        if not target_stretched_dir.exists():
            continue

        reference_csv = reference_dir_path / f"{block_name}.csv"
        if not reference_csv.exists():
            print(f"[SKIP] Missing reference CSV: {reference_csv}")
            total_skipped += 1
            continue

        name_target_map = _load_name_target_map(reference_csv)

        out_block = output_root_path / block_name
        out_target = out_block / "target"
        out_interference = out_block / "interference"
        out_mixture = out_block / "mixture"
        out_target.mkdir(parents=True, exist_ok=True)
        out_interference.mkdir(parents=True, exist_ok=True)
        out_mixture.mkdir(parents=True, exist_ok=True)

        wav_paths = sorted(target_stretched_dir.glob(f"{block_name}_stimulus*.wav"))
        print(f"\n=== Building final stimulus for {block_name} ({len(wav_paths)} files) ===")

        for target_wav_path in wav_paths:
            m = re.match(rf"{re.escape(block_name)}_stimulus(\d+)\.wav$", target_wav_path.name)
            if not m:
                continue

            num = int(m.group(1))
            try:
                num_interference = _interference_num(num)
            except ValueError as e:
                print(f"[SKIP] {target_wav_path.name}: {e}")
                total_skipped += 1
                continue

            name_target = name_target_map.get(num)
            if not name_target:
                print(f"[SKIP] No name_target for stimulus_num={num} in {reference_csv}")
                total_skipped += 1
                continue

            precue_path = precue_dir_path / f"{name_target}_cue.wav"
            interference_src_path = target_stretched_dir / f"{block_name}_stimulus{num_interference:02d}.wav"

            if not precue_path.exists():
                print(f"[SKIP] Missing precue: {precue_path}")
                total_skipped += 1
                continue
            if not interference_src_path.exists():
                print(f"[SKIP] Missing interference source: {interference_src_path}")
                total_skipped += 1
                continue

            target_audio, sr = _read_mono(target_wav_path)
            precue_audio, precue_sr = _read_mono(precue_path)
            interference_audio, inter_sr = _read_mono(interference_src_path)

            precue_audio = _resample_if_needed(precue_audio, precue_sr, sr)
            interference_audio = _resample_if_needed(interference_audio, inter_sr, sr)

            # Set target/interference energy ratio to 0 dB (equal power).
            eps = 1e-12
            target_power = float(np.mean(target_audio ** 2) + eps)
            interference_power = float(np.mean(interference_audio ** 2) + eps)
            gain = np.sqrt(target_power / interference_power)
            interference_audio = (interference_audio * gain).astype(np.float32)

            # 1) precue length to 4 sec (pad/truncate), 2) prepend to target.
            precue_fixed = _fit_to_seconds(precue_audio, sr, precue_sec)
            target_final = np.concatenate([precue_fixed, target_audio]).astype(np.float32)

            # Interference uses mapped stimulus and 4 sec zero padding at front.
            inter_front_zeros = np.zeros(int(round(precue_sec * sr)), dtype=np.float32)
            interference_final = np.concatenate([inter_front_zeros, interference_audio]).astype(np.float32)

            # Ensure same length before summation.
            final_len = max(len(target_final), len(interference_final))
            if len(target_final) < final_len:
                target_final = np.pad(target_final, (0, final_len - len(target_final)), mode="constant", constant_values=0.0)
            if len(interference_final) < final_len:
                interference_final = np.pad(interference_final, (0, final_len - len(interference_final)), mode="constant", constant_values=0.0)

            mixture = (target_final + interference_final).astype(np.float32)

            out_target_path = out_target / f"{block_name}_stimulus{num:02d}.wav"
            # B-rule: interference filename follows mixture/target index (num).
            out_interference_path = out_interference / f"{block_name}_stimulus{num:02d}.wav"
            out_mixture_path = out_mixture / f"{block_name}_stimulus{num:02d}.wav"

            sf.write(str(out_target_path), target_final, sr, subtype="PCM_16")
            sf.write(str(out_interference_path), interference_final, sr, subtype="PCM_16")
            sf.write(str(out_mixture_path), mixture, sr, subtype="PCM_16")
            total_done += 1

    print(f"\nDone. built={total_done}, skipped={total_skipped}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stretch WAV files to target duration.")
    parser.add_argument("--mode", choices=["single", "stimulus_eng", "stimulus_eng_final"], default="single")
    parser.add_argument("--input-dir", default="outputs/stimulus")
    parser.add_argument("--output-dir", default="outputs/stimulus_stretched")
    parser.add_argument("--target-sec", type=float, default=30.0)
    parser.add_argument("--info-csv", default="data/generated/stretch_info.csv")
    parser.add_argument("--root-dir", default="outputs/stimulus_eng")
    parser.add_argument("--precue-dir", default="outputs/precue_eng")
    parser.add_argument("--final-output-root", default="outputs/stimulus_eng_final")
    parser.add_argument("--reference-dir", default="data/reference_eng")
    parser.add_argument("--precue-sec", type=float, default=4.0)
    args = parser.parse_args()

    if args.mode == "stimulus_eng":
        process_stimulus_eng_root(root_dir=args.root_dir, target_sec=args.target_sec)
    elif args.mode == "stimulus_eng_final":
        process_stimulus_eng_final_root(
            stim_root=args.root_dir,
            precue_dir=args.precue_dir,
            output_root=args.final_output_root,
            reference_dir=args.reference_dir,
            precue_sec=args.precue_sec,
        )
    else:
        main(args.input_dir, args.output_dir, args.target_sec, args.info_csv)
