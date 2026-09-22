#!/usr/bin/env python3
import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import wave

from generate_speech import fetch_audio_bytes_from_tts, read_voice_map


FILE_PATTERN = re.compile(r"day(\d+)block(\d+)\.csv$")


def get_wav_duration_sec(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def discover_input_csvs(input_dir: Path) -> List[Tuple[int, int, Path]]:
    found: List[Tuple[int, int, Path]] = []
    for csv_path in sorted(input_dir.glob("day*block*.csv")):
        m = FILE_PATTERN.match(csv_path.name)
        if not m:
            continue
        day = int(m.group(1))
        block = int(m.group(2))
        found.append((day, block, csv_path))
    return found


def is_before_checkpoint(
    day: int,
    block: int,
    stimulus_num: int,
    start_day: int,
    start_block: int,
    start_stimulus: int,
) -> bool:
    current = (day, block, stimulus_num)
    checkpoint = (start_day, start_block, start_stimulus)
    return current < checkpoint


def process_one_csv(
    csv_path: Path,
    day: int,
    block: int,
    voice_map: Dict[str, str],
    token: str,
    output_root: Path,
    low_sec: float,
    high_sec: float,
    timeout: float,
    start_day: int,
    start_block: int,
    start_stimulus: int,
    overwrite_existing: bool,
) -> Tuple[int, int, int, List[Dict[str, str]]]:
    required_cols = {"stimulus_num", "name_target", "utterance"}
    outliers: List[Dict[str, str]] = []

    target_dir = output_root / f"day{day}block{block}" / "target_raw"
    target_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    success = 0
    skipped = 0

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not required_cols.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"Missing required columns in {csv_path}. "
                f"Required: {required_cols}, actual: {reader.fieldnames}"
            )

        for row in reader:
            total += 1
            stim = (row.get("stimulus_num") or "").strip()
            name = (row.get("name_target") or "").strip()
            text = (row.get("utterance") or "").strip()

            if not stim or not name or not text:
                print(
                    f"[SKIP] day{day}block{block} missing value(s): "
                    f"stimulus_num={stim!r}, name_target={name!r}, utterance_len={len(text)}"
                )
                skipped += 1
                continue

            voice_id = voice_map.get(name)
            if not voice_id:
                print(f"[SKIP] day{day}block{block} no voice_id mapping for name_target='{name}'")
                skipped += 1
                continue

            try:
                stim_int = int(stim)
            except ValueError:
                print(f"[SKIP] day{day}block{block} invalid stimulus_num={stim!r}")
                skipped += 1
                continue
            out_name = f"day{day}block{block}_stimulus{stim_int:02d}.wav"
            out_path = target_dir / out_name

            if is_before_checkpoint(
                day=day,
                block=block,
                stimulus_num=stim_int,
                start_day=start_day,
                start_block=start_block,
                start_stimulus=start_stimulus,
            ):
                skipped += 1
                continue

            if out_path.exists() and not overwrite_existing:
                print(f"[SKIP] Already exists: {out_path}")
                skipped += 1
                continue

            try:
                audio = fetch_audio_bytes_from_tts(
                    token=token,
                    voice_id=voice_id,
                    text=text,
                    timeout=timeout,
                )
                with out_path.open("wb") as wf:
                    wf.write(audio)
                duration = get_wav_duration_sec(out_path)
                success += 1
                print(f"[OK] {out_path} ({duration:.2f}s)")

                if duration < low_sec or duration > high_sec:
                    outliers.append(
                        {
                            "day": str(day),
                            "block": str(block),
                            "stimulus_num": stim,
                            "name_target": name,
                            "duration_sec": f"{duration:.4f}",
                            "wav_path": str(out_path),
                        }
                    )
            except Exception as e:
                print(f"[FAIL] day{day}block{block} stimulus={stim} name_target={name} -> {e}")

    return total, success, skipped, outliers


def save_outlier_log(rows: List[Dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["day", "block", "stimulus_num", "name_target", "duration_sec", "wav_path"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate English target_raw speech from data/reference_eng/day{i}block{j}.csv"
    )
    parser.add_argument(
        "--input-dir",
        default="data/reference_eng",
        help="Directory containing day{i}block{j}.csv files",
    )
    parser.add_argument(
        "--voice-info",
        default="data/generated/voice_info.csv",
        help="Path to voice_info CSV (name -> voice_id)",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/stimulus_eng",
        help="Root output directory",
    )
    parser.add_argument(
        "--outlier-log",
        default="data/generated/stimulus_eng_out_of_range.csv",
        help="CSV path to log generated samples outside duration range",
    )
    parser.add_argument("--min-sec", type=float, default=25.0, help="Minimum acceptable duration")
    parser.add_argument("--max-sec", type=float, default=35.0, help="Maximum acceptable duration")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout seconds")
    parser.add_argument("--start-day", type=int, default=1, help="Start from this day (inclusive)")
    parser.add_argument("--start-block", type=int, default=1, help="Start from this block (inclusive)")
    parser.add_argument(
        "--start-stimulus",
        type=int,
        default=1,
        help="Start from this stimulus number within the start day/block (inclusive)",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite existing WAV files (default: skip existing files)",
    )
    args = parser.parse_args()

    token = os.environ.get("TYPECAST_API_TOKEN")
    if not token:
        print("ERROR: TYPECAST_API_TOKEN is not set.", file=sys.stderr)
        sys.exit(1)

    input_dir = Path(args.input_dir)
    voice_info_path = Path(args.voice_info)
    output_root = Path(args.output_root)
    outlier_log = Path(args.outlier_log)

    if not input_dir.exists():
        print(f"ERROR: input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)
    if not voice_info_path.exists():
        print(f"ERROR: voice_info CSV not found: {voice_info_path}", file=sys.stderr)
        sys.exit(1)

    tasks = discover_input_csvs(input_dir)
    if not tasks:
        print(f"ERROR: no day{{i}}block{{j}}.csv files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    voice_map = read_voice_map(voice_info_path)

    total_all = 0
    success_all = 0
    skipped_all = 0
    outliers_all: List[Dict[str, str]] = []

    for day, block, csv_path in tasks:
        print(f"\n=== Processing day{day}block{block} ===")
        total, success, skipped, outliers = process_one_csv(
            csv_path=csv_path,
            day=day,
            block=block,
            voice_map=voice_map,
            token=token,
            output_root=output_root,
            low_sec=args.min_sec,
            high_sec=args.max_sec,
            timeout=args.timeout,
            start_day=args.start_day,
            start_block=args.start_block,
            start_stimulus=args.start_stimulus,
            overwrite_existing=args.overwrite_existing,
        )
        total_all += total
        success_all += success
        skipped_all += skipped
        outliers_all.extend(outliers)

    save_outlier_log(outliers_all, outlier_log)
    print("\n=== Done ===")
    print(f"Total rows: {total_all}")
    print(f"Successful generations: {success_all}")
    print(f"Skipped rows: {skipped_all}")
    print(f"Out-of-range samples (<{args.min_sec}s or >{args.max_sec}s): {len(outliers_all)}")
    print(f"Outlier log saved: {outlier_log.resolve()}")


if __name__ == "__main__":
    main()
