#!/usr/bin/env python3
"""
Generate English precue WAV files using Typecast.

Defaults:
- Text: "Please focus on this voice"
- Output directory: outputs/precue_eng
- Input voice list: data/reference/typecast_ko_standard_duration.csv
"""

import argparse
import os
import sys
import time
from pathlib import Path

from generate_pre_cue import boolish, pick, read_rows, sanitize_name, save_wav_bytes, tts_once


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate English precue WAV files.")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("TYPECAST_API_TOKEN"),
        help="Typecast X-API-KEY; or set env TYPECAST_API_TOKEN",
    )
    parser.add_argument(
        "--in-file",
        default="data/reference/typecast_ko_standard_duration.csv",
        help="Input CSV or JSON path",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model override (e.g., ssfm-v30). If absent, use per-row model or API default.",
    )
    parser.add_argument(
        "--text",
        default="Please focus on this voice",
        help="English cue text",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Exact duration in seconds",
    )
    parser.add_argument(
        "--dir",
        default="outputs/precue_eng",
        help="Output directory",
    )
    parser.add_argument(
        "--only-supported",
        action="store_true",
        help="Use rows with supports_kor==True and supports_duration==True (JSON only)",
    )
    parser.add_argument(
        "--force-duration",
        action="store_true",
        help="Force sending duration even if supports_duration is False/unknown",
    )
    parser.add_argument("--sleep", type=float, default=0.1, help="Sleep between API calls")
    parser.add_argument("--timeout", type=float, default=40.0, help="HTTP timeout")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        print("ERROR: Provide --api-key or set TYPECAST_API_TOKEN", file=sys.stderr)
        sys.exit(2)

    output_dir = Path(args.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.in_file)
    total = len(rows)
    print(f"[i] Rows: {total}")

    made = 0
    skipped = 0
    for row in rows:
        voice_id = pick(row, "voice_id", "id", "voiceId", "uuid")
        voice_name = pick(row, "voice_name", "name", "title", "display_name", "label") or ""
        model = args.model or pick(row, "model")

        if not voice_id:
            skipped += 1
            continue

        if args.only_supported:
            sup_k = boolish(row.get("supports_kor"))
            sup_d = boolish(row.get("supports_duration"))
            if sup_k is False or sup_k is None or sup_d is False or sup_d is None:
                skipped += 1
                continue

        if not args.force_duration:
            sup_d = boolish(row.get("supports_duration"))
            if sup_d is False:
                skipped += 1
                continue

        safe_name = sanitize_name(voice_name, fallback=voice_id)
        out_path = output_dir / f"{safe_name}_cue.wav"

        wav = tts_once(
            args.api_key,
            voice_id,
            args.text,
            args.duration,
            model,
            args.timeout,
            args.verbose,
        )
        if wav is None:
            skipped += 1
            continue

        save_wav_bytes(wav, str(out_path))
        made += 1
        print(f"[✓] {out_path}")
        time.sleep(args.sleep)

    print(f"[done] made={made}, skipped={skipped}, total={total}")


if __name__ == "__main__":
    main()
