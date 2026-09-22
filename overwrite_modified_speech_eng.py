#!/usr/bin/env python3
import argparse
import csv
import os
import re
import shutil
import sys
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from generate_speech import fetch_audio_bytes_from_tts, read_voice_map


BLOCK_PATTERN = re.compile(r"^day(\d+)block(\d+)\.csv$")
OUTLIER_REQUIRED = {"day", "block", "stimulus_num"}


def get_wav_duration_sec(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def load_csv_by_stimulus(csv_path: Path) -> Dict[str, Dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {"stimulus_num", "name_target", "utterance"}
        if not required.issubset(fieldnames):
            raise ValueError(
                f"Missing required columns in {csv_path}. "
                f"Required: {required}, actual: {reader.fieldnames}"
            )
        out: Dict[str, Dict[str, str]] = {}
        for row in reader:
            stim = (row.get("stimulus_num") or "").strip()
            if stim:
                out[stim] = row
        return out


def parse_outlier_targets(outlier_csv: Path) -> Dict[Tuple[int, int], List[int]]:
    grouped: Dict[Tuple[int, int], List[int]] = {}
    with outlier_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not OUTLIER_REQUIRED.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"Missing required columns in outlier CSV. "
                f"Required: {OUTLIER_REQUIRED}, actual: {reader.fieldnames}"
            )
        for row in reader:
            try:
                day = int((row.get("day") or "").strip())
                block = int((row.get("block") or "").strip())
                stim = int((row.get("stimulus_num") or "").strip())
            except ValueError:
                continue
            grouped.setdefault((day, block), []).append(stim)
    for k in grouped:
        grouped[k] = sorted(set(grouped[k]))
    return grouped


def backup_existing_wav_if_needed(
    wav_path: Path,
    backup_root: Path,
    day: int,
    block: int,
) -> Optional[Path]:
    if not wav_path.exists():
        return None
    dst = backup_root / f"day{day}block{block}" / "target_raw" / wav_path.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(wav_path, dst)
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overwrite English target_raw TTS outputs using stimulus_eng_out_of_range.csv targets."
    )
    parser.add_argument(
        "--reference-dir",
        default="data/reference_eng",
        help="Directory with current day{i}block{j}.csv files",
    )
    parser.add_argument(
        "--outlier-csv",
        default="data/generated/stimulus_eng_out_of_range.csv",
        help="Outlier list CSV (day, block, stimulus_num)",
    )
    parser.add_argument(
        "--voice-info",
        default="data/generated/voice_info.csv",
        help="Path to voice_info CSV",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/stimulus_eng",
        help="Root output directory",
    )
    parser.add_argument(
        "--tts-backup-root",
        default=None,
        help="Backup root for existing WAV files before overwrite "
        "(default: outputs/stimulus_eng/_backup_overwrite/<timestamp>)",
    )
    parser.add_argument(
        "--log-csv",
        default="data/generated/stimulus_eng_overwrite_log.csv",
        help="Log CSV path",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="TTS HTTP timeout")
    parser.add_argument("--sleep", type=float, default=0.05, help="Sleep seconds between TTS calls")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without writing files")
    args = parser.parse_args()

    token = os.environ.get("TYPECAST_API_TOKEN")
    if not token:
        print("ERROR: TYPECAST_API_TOKEN is not set.", file=sys.stderr)
        sys.exit(1)

    reference_dir = Path(args.reference_dir)
    outlier_csv = Path(args.outlier_csv)
    voice_info_path = Path(args.voice_info)
    output_root = Path(args.output_root)
    log_csv = Path(args.log_csv)

    if not reference_dir.exists():
        print(f"ERROR: reference directory not found: {reference_dir}", file=sys.stderr)
        sys.exit(1)
    if not outlier_csv.exists():
        print(f"ERROR: outlier CSV not found: {outlier_csv}", file=sys.stderr)
        sys.exit(1)
    if not voice_info_path.exists():
        print(f"ERROR: voice_info CSV not found: {voice_info_path}", file=sys.stderr)
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.tts_backup_root:
        tts_backup_root = Path(args.tts_backup_root)
    else:
        tts_backup_root = output_root / "_backup_overwrite" / ts

    voice_map = read_voice_map(voice_info_path)

    target_map = parse_outlier_targets(outlier_csv)
    if not target_map:
        print("ERROR: no valid targets found in outlier CSV.", file=sys.stderr)
        sys.exit(1)

    log_rows: List[Dict[str, str]] = []
    total_targets = 0
    total_overwritten = 0

    for (day, block), stim_list in sorted(target_map.items()):
        block_stem = f"day{day}block{block}"
        current_csv = reference_dir / f"{block_stem}.csv"
        if not current_csv.exists():
            print(f"[WARN] Missing source CSV for {block_stem}, skipping.")
            continue

        row_map = load_csv_by_stimulus(current_csv)
        print(f"\n=== {block_stem}: {len(stim_list)} target stimulus rows ===")
        total_targets += len(stim_list)

        target_dir = output_root / block_stem / "target_raw"
        target_dir.mkdir(parents=True, exist_ok=True)

        for stim_int in stim_list:
            stim = str(stim_int)
            row = row_map.get(stim)
            if row is None:
                print(f"[SKIP] Missing stimulus row in CSV: {block_stem} stimulus={stim}")
                continue

            name = (row.get("name_target") or "").strip()
            text = (row.get("utterance") or "").strip()
            if not stim or not name or not text:
                print(f"[SKIP] Missing value in {block_stem} stimulus={stim!r}")
                continue

            voice_id = voice_map.get(name)
            if not voice_id:
                print(f"[SKIP] No voice mapping for name_target='{name}' ({block_stem} stimulus={stim})")
                continue

            wav_name = f"{block_stem}_stimulus{stim_int:02d}.wav"
            wav_path = target_dir / wav_name

            backup_wav_path = backup_existing_wav_if_needed(
                wav_path=wav_path,
                backup_root=tts_backup_root,
                day=day,
                block=block,
            )

            if args.dry_run:
                print(f"[DRY-RUN] Would overwrite: {wav_path}")
                log_rows.append(
                    {
                        "day": str(day),
                        "block": str(block),
                        "stimulus_num": stim,
                        "name_target": name,
                        "status": "planned",
                        "wav_path": str(wav_path),
                        "backup_wav_path": str(backup_wav_path) if backup_wav_path else "",
                        "duration_sec": "",
                    }
                )
                continue

            try:
                audio = fetch_audio_bytes_from_tts(
                    token=token,
                    voice_id=voice_id,
                    text=text,
                    timeout=args.timeout,
                )
                with wav_path.open("wb") as f:
                    f.write(audio)
                dur = get_wav_duration_sec(wav_path)
                total_overwritten += 1
                print(f"[OK] Overwritten: {wav_path} ({dur:.2f}s)")
                log_rows.append(
                    {
                        "day": str(day),
                        "block": str(block),
                        "stimulus_num": stim,
                        "name_target": name,
                        "status": "overwritten",
                        "wav_path": str(wav_path),
                        "backup_wav_path": str(backup_wav_path) if backup_wav_path else "",
                        "duration_sec": f"{dur:.4f}",
                    }
                )
            except Exception as e:
                print(f"[FAIL] {block_stem} stimulus={stim} name_target={name} -> {e}")
                log_rows.append(
                    {
                        "day": str(day),
                        "block": str(block),
                        "stimulus_num": stim,
                        "name_target": name,
                        "status": f"failed: {e}",
                        "wav_path": str(wav_path),
                        "backup_wav_path": str(backup_wav_path) if backup_wav_path else "",
                        "duration_sec": "",
                    }
                )
            if args.sleep > 0:
                time.sleep(args.sleep)

    log_csv.parent.mkdir(parents=True, exist_ok=True)
    with log_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "day",
                "block",
                "stimulus_num",
                "name_target",
                "status",
                "wav_path",
                "backup_wav_path",
                "duration_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(log_rows)

    print("\n=== Done ===")
    print(f"Target rows from outlier CSV: {total_targets}")
    print(f"Overwritten WAV files: {total_overwritten}")
    if not args.dry_run:
        print(f"TTS overwrite backup root: {tts_backup_root.resolve()}")
    print(f"Log CSV: {log_csv.resolve()}")


if __name__ == "__main__":
    main()
