#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make 2s cues from Typecast voices listed in a CSV/JSON.

- Input: CSV or JSON written earlier (must contain at least voice_id, voice_name; optionally model, supports_kor, supports_duration)
- Output: {voice_name}_cue.wav (non-alphabet characters are removed from voice_name)
- Synthesis: POST https://api.typecast.ai/v1/text-to-speech (X-API-KEY header)
- Duration: exact 2 seconds via request body: duration=2

Usage:
  export TYPECAST_API_KEY="YOUR_KEY"
  python make_cues_2s.py --in-file typecast_voices.json --only-supported
  # or CSV
  python make_cues_2s.py --in-file typecast_voices.csv --only-supported

Notes:
- If --only-supported is enabled, only rows with supports_kor==True and supports_duration==True are used (JSON only).
- If response is binary WAV, save directly. If response is JSON with "audio_download_url", download from that URL.
- Voices without duration support may return 4xx (usually 422) and are skipped by default. Use --force-duration to ignore.
"""

import argparse
import csv
import io
import json
import os
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

import requests

API_BASE = "https://api.typecast.ai"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", default=os.environ.get("TYPECAST_API_TOKEN"),
                   help="Typecast X-API-KEY; or set env TYPECAST_API_TOKEN")
    p.add_argument("--in-file", default='data/reference/typecast_ko_standard_duration.csv', help="Input CSV or JSON path")
    p.add_argument("--model", default=None, help="Model override (e.g., ssfm-v30). If absent, uses per-row model or API default.")
    p.add_argument("--text", default="이 목소리에 집중해주세요", help="Synthesis text (default Korean cue)")
    p.add_argument("--duration", type=float, default=2.0, help="Exact duration in seconds (default 2.0)")
    p.add_argument("--dir", default="outputs/precue", help="Output directory")
    p.add_argument("--only-supported", action="store_true",
                   help="Use rows with supports_kor==True AND supports_duration==True (JSON only)")
    p.add_argument("--force-duration", action="store_true",
                   help="Force sending duration even if supports_duration flag is False/unknown")
    p.add_argument("--sleep", type=float, default=0.1, help="Sleep between calls to be gentle on API")
    p.add_argument("--timeout", type=float, default=40.0, help="HTTP timeout")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

def H(key: str) -> Dict[str, str]:
    return {
        "X-API-KEY": key,
        "Accept": "application/json, audio/wav",
        "Content-Type": "application/json",
    }

def read_rows(path: str) -> List[Dict[str, Any]]:
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "voices" in data and isinstance(data["voices"], list):
            return data["voices"]
        # fallback: first list
        for v in (data.values() if isinstance(data, dict) else []):
            if isinstance(v, list):
                return v
        raise ValueError("JSON shape not recognized.")
    elif path.lower().endswith(".csv"):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                out.append(row)
        return out
    else:
        raise ValueError("Unsupported file extension. Use .json or .csv")

def sanitize_name(name: Optional[str], fallback: str) -> str:
    base = (name or "").strip()
    if not base:
        base = fallback
    # Keep alphabetic characters only.
    safe = re.sub(r"[^A-Za-z]+", "", base)
    if not safe:
        safe = re.sub(r"[^A-Za-z0-9]+", "", fallback) or "voice"
    return safe

def pick(row: Dict[str, Any], *keys) -> Optional[str]:
    for k in keys:
        if k in row and isinstance(row[k], str) and row[k].strip():
            return row[k].strip()
    return None

def boolish(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v.lower() in ("true","1","yes","y"):
            return True
        if v.lower() in ("false","0","no","n"):
            return False
    return None

def save_wav_bytes(b: bytes, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(b)

def fetch_bytes(url: str, headers: Dict[str,str], timeout: float) -> bytes:
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.content

def tts_once(api_key: str, voice_id: str, text: str, duration_s: float, model: Optional[str], timeout: float, verbose: bool) -> Optional[bytes]:
    url = f"{API_BASE}/v1/text-to-speech"
    body: Dict[str, Any] = {
        "voice_id": voice_id,
        "text": text,
        # "language": "kor",
        "output": {
            "audio_format": "wav",
            "audio_tempo": 1,
            "audio_pitch": 0,
            "volume": 100
        },
        "prompt": {
            "emotion_preset": "normal",
            "emotion_intensity": 1
        },
        "seed": 42,
        "model": 'ssfm-v30'
    }
    if model:
        body["model"] = model
    # Lock duration to a fixed value.
    body["duration"] = float(duration_s)

    if verbose:
        print(f"[POST] {url} voice={voice_id} duration={duration_s} model={model}")

    r = requests.post(url, headers=H(api_key), json=body, timeout=timeout)
    # Some environments return JSON metadata/URL, others return binary audio directly.
    ct = r.headers.get("Content-Type","").lower()
    if r.status_code == 200 and "audio" in ct:
        return r.content
    # 200 with JSON response (for example with downloadable URL).
    if r.status_code == 200 and "json" in ct:
        try:
            data = r.json()
        except Exception:
            data = {}
        # Handle common URL field names.
        for k in ("audio_download_url", "audioUrl", "url", "download_url"):
            val = data.get(k)
            if isinstance(val, str) and val.startswith("http"):
                return fetch_bytes(val, H(api_key), timeout)
        # If data contains inline base64/bytes, add handling here depending on API behavior.
    # Return None on 4xx (for example, unsupported duration).
    if r.status_code >= 400:
        if verbose:
            try:
                print(f"[WARN] {voice_id} -> {r.status_code} {r.text[:300]}", file=sys.stderr)
            except Exception:
                print(f"[WARN] {voice_id} -> {r.status_code}", file=sys.stderr)
        return None
    # Other cases: raise.
    r.raise_for_status()
    return None

def main():
    args = parse_args()
    if not args.api_key:
        print("ERROR: Provide --api-key or set TYPECAST_API_KEY", file=sys.stderr)
        sys.exit(2)

    rows = read_rows(args.in_file)
    total = len(rows)
    print(f"[i] Rows: {total}")

    made = 0
    skipped = 0
    for i, row in enumerate(rows, 1):
        voice_id = pick(row, "voice_id", "id", "voiceId", "uuid")
        voice_name = pick(row, "voice_name", "name", "title", "display_name", "label") or ""
        model = args.model or pick(row, "model")

        if not voice_id:
            skipped += 1
            continue

        # When --only-supported is enabled, JSON must have both supports_kor and supports_duration as True.
        if args.only_supported:
            sup_k = boolish(row.get("supports_kor"))
            sup_d = boolish(row.get("supports_duration"))
            if sup_k is False or sup_k is None or sup_d is False or sup_d is None:
                skipped += 1
                continue
        # If --force-duration is not set and supports_duration is explicitly False, skip.
        if not args.force_duration:
            sup_d = boolish(row.get("supports_duration"))
            if sup_d is False:
                skipped += 1
                continue

        safe_name = sanitize_name(voice_name, fallback=voice_id)
        out_path = os.path.join(args.dir, f"{safe_name}_cue.wav")

        wav = tts_once(args.api_key, voice_id, args.text, args.duration, model, args.timeout, args.verbose)
        if wav is None:
            skipped += 1
            continue

        save_wav_bytes(wav, out_path)
        made += 1
        print(f"[✓] {out_path}")
        time.sleep(args.sleep)

    print(f"[done] made={made}, skipped={skipped}, total={total}")

if __name__ == "__main__":
    main()
