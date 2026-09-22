#!/usr/bin/env python3
"""
Find target_raw WAVs outside the scan duration window, then re-synthesize with
Typecast audio_tempo until each file lands in the save window (default 25–30 s).

Example:
  python regenerate_stimulus_eng_by_tempo.py --root-dir outputs/stimulus_eng_d2
"""
from __future__ import annotations

import argparse
import csv
import io
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


WAV_NAME_RE = re.compile(
    r"^day(?P<day>\d+)block(?P<block>\d+)_stimulus(?P<stim>\d+)\.wav$",
    re.IGNORECASE,
)
TEMPO_MIN = 0.5
TEMPO_MAX = 2.0


def get_wav_duration_sec(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def duration_from_wav_bytes(audio: bytes) -> float:
    with wave.open(io.BytesIO(audio), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def clamp_tempo(tempo: float) -> float:
    return float(min(TEMPO_MAX, max(TEMPO_MIN, tempo)))


def next_tempo(current_tempo: float, duration_sec: float, target_sec: float) -> float:
    """
    Scale tempo so estimated duration moves toward target_sec.
    Longer audio -> higher tempo (faster); shorter -> lower tempo (slower).
    """
    if duration_sec <= 0 or target_sec <= 0:
        return clamp_tempo(current_tempo)
    proposed = current_tempo * (duration_sec / target_sec)
    proposed = clamp_tempo(proposed)
    # If clamping left tempo unchanged, nudge toward the needed direction.
    if abs(proposed - current_tempo) < 1e-6:
        if duration_sec > target_sec:
            proposed = clamp_tempo(current_tempo * 1.05)
        else:
            proposed = clamp_tempo(current_tempo * 0.95)
    return proposed


def load_reference_row(
    reference_dir: Path, day: int, block: int, stimulus_num: int
) -> Optional[Tuple[str, str]]:
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


def discover_outliers(
    root_dir: Path,
    scan_min_sec: float,
    scan_max_sec: float,
) -> List[Tuple[Path, int, int, int, float]]:
    """Return (wav_path, day, block, stimulus_num, duration) outside scan window."""
    found: List[Tuple[Path, int, int, int, float]] = []
    for wav_path in sorted(root_dir.glob("day*block*/target_raw/*.wav")):
        m = WAV_NAME_RE.match(wav_path.name)
        if not m:
            print(f"[WARN] Filename not matched, skip: {wav_path.name}")
            continue
        day = int(m.group("day"))
        block = int(m.group("block"))
        stim = int(m.group("stim"))
        try:
            dur = get_wav_duration_sec(wav_path)
        except Exception as e:
            print(f"[WARN] Could not read duration for {wav_path}: {e}")
            continue
        if dur < scan_min_sec or dur > scan_max_sec:
            found.append((wav_path, day, block, stim, dur))
    return found


def backup_wav(wav_path: Path, backup_root: Path, day: int, block: int) -> Path:
    dst = backup_root / f"day{day}block{block}" / "target_raw" / wav_path.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(wav_path, dst)
    return dst


def synthesize_until_in_range(
    token: str,
    voice_id: str,
    text: str,
    timeout: float,
    save_min_sec: float,
    save_max_sec: float,
    target_sec: float,
    max_attempts: int,
    sleep_s: float,
    initial_tempo: float = 1.0,
) -> Tuple[Optional[bytes], float, float, int, str]:
    """
    Returns (audio_bytes|None, final_duration, final_tempo, attempts_used, status).
    status: ok | skipped_max_attempts | failed
    """
    tempo = clamp_tempo(initial_tempo)
    last_dur = 0.0
    for attempt in range(1, max_attempts + 1):
        try:
            audio = fetch_audio_bytes_from_tts(
                token=token,
                voice_id=voice_id,
                text=text,
                timeout=timeout,
                audio_tempo=tempo,
            )
            dur = duration_from_wav_bytes(audio)
            last_dur = dur
            print(
                f"  attempt {attempt}/{max_attempts}: tempo={tempo:.4f} -> {dur:.2f}s"
            )
            if save_min_sec <= dur <= save_max_sec:
                return audio, dur, tempo, attempt, "ok"

            tempo = next_tempo(tempo, dur, target_sec)
        except Exception as e:
            print(f"  attempt {attempt}/{max_attempts}: TTS failed: {e}")
            if attempt == max_attempts:
                return None, last_dur, tempo, attempt, f"failed: {e}"
        if sleep_s > 0:
            time.sleep(sleep_s)

    return None, last_dur, tempo, max_attempts, "skipped_max_attempts"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a stimulus_eng root for target_raw WAVs outside the scan window, "
            "then re-synthesize with audio_tempo until duration is in the save window."
        )
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Stimulus root, e.g. outputs/stimulus_eng_d2",
    )
    parser.add_argument(
        "--reference-dir",
        default="data/reference_eng",
        help="Directory with day{i}block{j}.csv (default: data/reference_eng)",
    )
    parser.add_argument(
        "--voice-info",
        default="data/generated/voice_info.csv",
        help="name -> voice_id mapping CSV",
    )
    parser.add_argument(
        "--scan-min-sec",
        type=float,
        default=25.0,
        help="Existing WAV below this is treated as outlier (default: 25)",
    )
    parser.add_argument(
        "--scan-max-sec",
        type=float,
        default=35.0,
        help="Existing WAV above this is treated as outlier (default: 35)",
    )
    parser.add_argument(
        "--save-min-sec",
        type=float,
        default=25.0,
        help="Accept regenerated WAV only if duration >= this (default: 25)",
    )
    parser.add_argument(
        "--save-max-sec",
        type=float,
        default=35.0,
        help="Accept regenerated WAV only if duration <= this (default: 30)",
    )
    parser.add_argument(
        "--target-sec",
        type=float,
        default=27.5,
        help="Tempo estimation target duration in seconds (default: 27.5)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=10,
        help="Max TTS attempts per file before skip (default: 10)",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument(
        "--log-csv",
        default="data/generated/stimulus_eng_tempo_regen_log.csv",
        help="Result log CSV path",
    )
    parser.add_argument(
        "--backup-root",
        default=None,
        help="Backup root before overwrite "
        "(default: <root-dir>/_backup_tempo/<timestamp>)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List outliers only; do not call TTS or write files",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    reference_dir = Path(args.reference_dir)
    voice_info_path = Path(args.voice_info)
    log_csv = Path(args.log_csv)

    if not root_dir.is_dir():
        print(f"ERROR: root directory not found: {root_dir}", file=sys.stderr)
        sys.exit(1)
    if not reference_dir.is_dir():
        print(f"ERROR: reference directory not found: {reference_dir}", file=sys.stderr)
        sys.exit(1)
    if not voice_info_path.is_file():
        print(f"ERROR: voice_info CSV not found: {voice_info_path}", file=sys.stderr)
        sys.exit(1)
    if args.save_min_sec > args.save_max_sec:
        print("ERROR: --save-min-sec must be <= --save-max-sec", file=sys.stderr)
        sys.exit(1)

    token = os.environ.get("TYPECAST_API_TOKEN")
    if not args.dry_run and not token:
        print("ERROR: TYPECAST_API_TOKEN is not set.", file=sys.stderr)
        sys.exit(1)

    outliers = discover_outliers(root_dir, args.scan_min_sec, args.scan_max_sec)
    print(
        f"Found {len(outliers)} outlier(s) outside "
        f"[{args.scan_min_sec}, {args.scan_max_sec}] s under {root_dir}"
    )
    if not outliers:
        print("Nothing to regenerate.")
        return

    voice_map = read_voice_map(voice_info_path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = Path(args.backup_root) if args.backup_root else (root_dir / "_backup_tempo" / ts)

    log_rows: List[Dict[str, str]] = []
    ok_n = skip_n = fail_n = 0

    for wav_path, day, block, stim, old_dur in outliers:
        block_stem = f"day{day}block{block}"
        print(f"\n=== {wav_path.name} (was {old_dur:.2f}s) ===")

        ref = load_reference_row(reference_dir, day, block, stim)
        if ref is None:
            skip_n += 1
            log_rows.append(
                {
                    "day": str(day),
                    "block": str(block),
                    "stimulus_num": str(stim),
                    "name_target": "",
                    "old_duration_sec": f"{old_dur:.4f}",
                    "new_duration_sec": "",
                    "final_tempo": "",
                    "attempts": "0",
                    "status": "skipped_missing_reference",
                    "wav_path": str(wav_path),
                    "backup_wav_path": "",
                }
            )
            continue

        name, text = ref
        voice_id = voice_map.get(name)
        if not voice_id:
            print(f"[SKIP] No voice_id for name_target={name!r}")
            skip_n += 1
            log_rows.append(
                {
                    "day": str(day),
                    "block": str(block),
                    "stimulus_num": str(stim),
                    "name_target": name,
                    "old_duration_sec": f"{old_dur:.4f}",
                    "new_duration_sec": "",
                    "final_tempo": "",
                    "attempts": "0",
                    "status": "skipped_missing_voice",
                    "wav_path": str(wav_path),
                    "backup_wav_path": "",
                }
            )
            continue

        if args.dry_run:
            print(f"[DRY-RUN] Would regenerate {block_stem} stimulus={stim} name={name}")
            log_rows.append(
                {
                    "day": str(day),
                    "block": str(block),
                    "stimulus_num": str(stim),
                    "name_target": name,
                    "old_duration_sec": f"{old_dur:.4f}",
                    "new_duration_sec": "",
                    "final_tempo": "",
                    "attempts": "0",
                    "status": "planned",
                    "wav_path": str(wav_path),
                    "backup_wav_path": "",
                }
            )
            continue

        # Seed tempo from existing duration so the first attempt is closer.
        seed_tempo = next_tempo(1.0, old_dur, args.target_sec)
        print(f"  seed tempo from {old_dur:.2f}s -> {seed_tempo:.4f}")

        audio, new_dur, final_tempo, attempts, status = synthesize_until_in_range(
            token=token or "",
            voice_id=voice_id,
            text=text,
            timeout=args.timeout,
            save_min_sec=args.save_min_sec,
            save_max_sec=args.save_max_sec,
            target_sec=args.target_sec,
            max_attempts=args.max_attempts,
            sleep_s=args.sleep,
            initial_tempo=seed_tempo,
        )

        backup_path = ""
        if status == "ok" and audio is not None:
            bp = backup_wav(wav_path, backup_root, day, block)
            backup_path = str(bp)
            with wav_path.open("wb") as f:
                f.write(audio)
            ok_n += 1
            print(
                f"[OK] Saved {wav_path} ({new_dur:.2f}s, tempo={final_tempo:.4f}, "
                f"attempts={attempts})"
            )
        elif status == "skipped_max_attempts":
            skip_n += 1
            print(
                f"[SKIP] Still out of [{args.save_min_sec}, {args.save_max_sec}] "
                f"after {attempts} attempts (last={new_dur:.2f}s, tempo={final_tempo:.4f})"
            )
        else:
            fail_n += 1
            print(f"[FAIL] {wav_path.name}: {status}")

        log_rows.append(
            {
                "day": str(day),
                "block": str(block),
                "stimulus_num": str(stim),
                "name_target": name,
                "old_duration_sec": f"{old_dur:.4f}",
                "new_duration_sec": f"{new_dur:.4f}" if new_dur else "",
                "final_tempo": f"{final_tempo:.4f}",
                "attempts": str(attempts),
                "status": status,
                "wav_path": str(wav_path),
                "backup_wav_path": backup_path,
            }
        )

    log_csv.parent.mkdir(parents=True, exist_ok=True)
    with log_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "day",
                "block",
                "stimulus_num",
                "name_target",
                "old_duration_sec",
                "new_duration_sec",
                "final_tempo",
                "attempts",
                "status",
                "wav_path",
                "backup_wav_path",
            ],
        )
        writer.writeheader()
        writer.writerows(log_rows)

    print("\n=== Done ===")
    print(f"Outliers scanned: {len(outliers)}")
    print(f"Saved (in [{args.save_min_sec}, {args.save_max_sec}] s): {ok_n}")
    print(f"Skipped: {skip_n}")
    print(f"Failed: {fail_n}")
    if not args.dry_run and ok_n:
        print(f"Backup root: {backup_root.resolve()}")
    print(f"Log CSV: {log_csv.resolve()}")


if __name__ == "__main__":
    main()
