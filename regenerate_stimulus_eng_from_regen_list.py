#!/usr/bin/env python3
"""
Regenerate English target_raw WAVs only for (day, block, stimulus_num) listed in
data/regen_list.csv, using name_target and utterance from data/reference_eng_final.
"""
import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import wave

from generate_speech import fetch_audio_bytes_from_tts, read_voice_map


def get_wav_duration_sec(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def load_reference_row(
    reference_dir: Path, day: int, block: int, stimulus_num: int
) -> Optional[Tuple[str, str]]:
    """Return (name_target, utterance) for stimulus_num, or None if missing."""
    csv_path = reference_dir / f"day{day}block{block}.csv"
    if not csv_path.is_file():
        print(f"[SKIP] reference file not found: {csv_path}", file=sys.stderr)
        return None

    required = {"stimulus_num", "name_target", "utterance"}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"Missing required columns in {csv_path}. "
                f"Required: {required}, actual: {reader.fieldnames}"
            )
        for row in reader:
            stim_raw = (row.get("stimulus_num") or "").strip()
            if not stim_raw:
                continue
            try:
                stim_int = int(stim_raw)
            except ValueError:
                continue
            if stim_int != stimulus_num:
                continue
            name = (row.get("name_target") or "").strip()
            text = (row.get("utterance") or "").strip()
            if not name or not text:
                print(
                    f"[SKIP] day{day}block{block} stimulus {stimulus_num}: "
                    f"empty name_target or utterance",
                    file=sys.stderr,
                )
                return None
            return name, text

    print(
        f"[SKIP] day{day}block{block} stimulus_num={stimulus_num} not found in {csv_path}",
        file=sys.stderr,
    )
    return None


def read_regen_list(path: Path) -> List[Tuple[int, int, int]]:
    rows: List[Tuple[int, int, int]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        need = {"day", "block", "stimulus_num"}
        if not need.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"regen_list must have columns {need}, got {reader.fieldnames}"
            )
        for i, row in enumerate(reader, start=2):
            d = (row.get("day") or "").strip()
            b = (row.get("block") or "").strip()
            s = (row.get("stimulus_num") or "").strip()
            if not d and not b and not s:
                continue
            try:
                rows.append((int(d), int(b), int(s)))
            except ValueError:
                raise ValueError(f"Invalid integers at {path}:{i}: day={d!r} block={b!r} stimulus_num={s!r}")
    return rows


def save_outlier_log(rows: List[Dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["day", "block", "stimulus_num", "name_target", "duration_sec", "wav_path"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate English stimuli listed in data/regen_list.csv "
        "(reference: data/reference_eng_final, output like generate_speech_eng.py)."
    )
    parser.add_argument(
        "--regen-list",
        default="data/regen_list.csv",
        help="CSV with columns day,block,stimulus_num",
    )
    parser.add_argument(
        "--reference-dir",
        default="data/reference_eng",
        help="Directory with day{i}block{j}.csv",
    )
    parser.add_argument(
        "--voice-info",
        default="data/generated/voice_info.csv",
        help="name -> voice_id mapping CSV",
    )
    parser.add_argument(
        "--output-root",
        default="output/stimlus_eng_regen",
        help="Root output (dayXblockY/target_raw/dayXblockY_stimulusNN.wav)",
    )
    parser.add_argument(
        "--outlier-log",
        default="data/generated/stimulus_eng_regen_out_of_range.csv",
        help="Log WAVs outside duration range",
    )
    parser.add_argument("--min-sec", type=float, default=25.0)
    parser.add_argument("--max-sec", type=float, default=35.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite existing WAV files",
    )
    args = parser.parse_args()

    token = os.environ.get("TYPECAST_API_TOKEN")
    if not token:
        print("ERROR: TYPECAST_API_TOKEN is not set.", file=sys.stderr)
        sys.exit(1)

    regen_path = Path(args.regen_list)
    reference_dir = Path(args.reference_dir)
    voice_info_path = Path(args.voice_info)
    output_root = Path(args.output_root)

    if not regen_path.is_file():
        print(f"ERROR: regen list not found: {regen_path}", file=sys.stderr)
        sys.exit(1)
    if not reference_dir.is_dir():
        print(f"ERROR: reference directory not found: {reference_dir}", file=sys.stderr)
        sys.exit(1)
    if not voice_info_path.is_file():
        print(f"ERROR: voice_info CSV not found: {voice_info_path}", file=sys.stderr)
        sys.exit(1)

    tasks = read_regen_list(regen_path)
    if not tasks:
        print("ERROR: no rows in regen list.", file=sys.stderr)
        sys.exit(1)

    voice_map = read_voice_map(voice_info_path)

    success = 0
    skipped = 0
    failed = 0
    outliers: List[Dict[str, str]] = []

    for day, block, stimulus_num in tasks:
        ref = load_reference_row(reference_dir, day, block, stimulus_num)
        if ref is None:
            skipped += 1
            continue
        name, text = ref
        voice_id = voice_map.get(name)
        if not voice_id:
            print(
                f"[SKIP] day{day}block{block} stimulus {stimulus_num}: "
                f"no voice_id for name_target={name!r}"
            )
            skipped += 1
            continue

        target_dir = output_root / f"day{day}block{block}" / "target_raw"
        target_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"day{day}block{block}_stimulus{stimulus_num:02d}.wav"
        out_path = target_dir / out_name

        if out_path.exists() and not args.overwrite_existing:
            print(f"[SKIP] Already exists: {out_path}")
            skipped += 1
            continue

        try:
            audio = fetch_audio_bytes_from_tts(
                token=token,
                voice_id=voice_id,
                text=text,
                timeout=args.timeout,
            )
            with out_path.open("wb") as wf:
                wf.write(audio)
            duration = get_wav_duration_sec(out_path)
            success += 1
            print(f"[OK] {out_path} ({duration:.2f}s)")

            if duration < args.min_sec or duration > args.max_sec:
                outliers.append(
                    {
                        "day": str(day),
                        "block": str(block),
                        "stimulus_num": str(stimulus_num),
                        "name_target": name,
                        "duration_sec": f"{duration:.4f}",
                        "wav_path": str(out_path),
                    }
                )
        except Exception as e:
            failed += 1
            print(
                f"[FAIL] day{day}block{block} stimulus={stimulus_num} name_target={name} -> {e}",
                file=sys.stderr,
            )

    outlier_path = Path(args.outlier_log)
    save_outlier_log(outliers, outlier_path)

    print("\n=== Done ===")
    print(f"Tasks from list: {len(tasks)}")
    print(f"Successful: {success}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(
        f"Out-of-range (<{args.min_sec}s or >{args.max_sec}s): {len(outliers)} "
        f"(log: {outlier_path.resolve()})"
    )


if __name__ == "__main__":
    main()
