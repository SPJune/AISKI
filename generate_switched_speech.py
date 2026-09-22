#!/usr/bin/env python3
import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import requests


TYPECAST_TTS_URL = "https://api.typecast.ai/v1/text-to-speech"

def H(key: str) -> Dict[str, str]:
    return {
        "X-API-KEY": key,
        "Accept": "application/json, audio/wav",
        "Content-Type": "application/json",
    }

def safe_filename(s: str) -> str:
    """
    Utility to sanitize a filename.
    Keep letters/numbers/Korean/space/._-, replace others with '_'.
    Collapse consecutive spaces into one underscore.
    """
    s = re.sub(r"[^\w\s.\-가-힣]", "_", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s.strip())
    return s or "unknown"


def read_voice_map(voice_info_path: Path) -> Dict[str, str]:
    """
    Load name -> voice_id mapping from voice_info CSV.
    """
    mapping: Dict[str, str] = {}
    with voice_info_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"name", "voice_id"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"Missing required headers in voice_info CSV. Required: {required}, actual: {reader.fieldnames}"
            )
        for row in reader:
            name = (row.get("name") or "").strip()
            vid = (row.get("voice_id") or "").strip()
            if name and vid:
                mapping[name] = vid
    if not mapping:
        raise ValueError("No valid (name, voice_id) mapping found in voice_info CSV.")
    return mapping


def fetch_audio_bytes_from_tts(
    token: str,
    voice_id: str,
    text: str,
    timeout: float = 60.0,
    max_retries: int = 5,
) -> bytes:
    """
    Call Typecast TTS and return audio bytes.
    Retries use exponential backoff for 429/5xx.
    """
    headers = H(token)
    '''
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        # Some APIs support direct audio via Accept header.
    }
    '''
    payload = {
        "voice_id": voice_id,
        "text": text,
        "seed": 42,
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
        "model": 'ssfm-v30'
        # "speed": 1.0,
        # "pitch": 0.0,
    }

    def _post_tts():
        return requests.post(TYPECAST_TTS_URL, headers=headers, json=payload, timeout=timeout)

    # Retry loop.
    for attempt in range(1, max_retries + 1):
        try:
            resp = _post_tts()
            # Retry on rate-limit/server errors.
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"Retryable status {resp.status_code}", response=resp)

            resp.raise_for_status()

            ctype = resp.headers.get("Content-Type", "").lower()

            # 1) Direct audio response.
            if "audio" in ctype or "octet-stream" in ctype:
                return resp.content

            # 2) JSON response with audio URL/path.
            # Try candidate keys.
            if "application/json" in ctype or ctype.endswith("+json"):
                data = resp.json()
                # Common key names.
                candidate_keys = [
                    "audio_url",
                    "url",
                    "result_url",
                    "audio",
                    "audioPath",
                    "file_url",
                    "wav_url",
                ]
                for k in candidate_keys:
                    if k in data and isinstance(data[k], str) and data[k].startswith(("http://", "https://")):
                        audio_url = data[k]
                        # Download audio bytes.
                        a = requests.get(audio_url, timeout=timeout)
                        a.raise_for_status()
                        return a.content

                # Handle inline base64 audio when provided.
                if "audio_base64" in data and isinstance(data["audio_base64"], str):
                    import base64
                    return base64.b64decode(data["audio_base64"])

                raise RuntimeError(
                    f"Could not find audio URL/data in TTS JSON response. keys={list(data.keys())}"
                )

            # Unexpected content type.
            raise RuntimeError(f"Unexpected Content-Type: {ctype}")

        except (requests.Timeout, requests.ConnectionError) as e:
            # Retry transient network errors.
            if attempt == max_retries:
                raise
        except requests.HTTPError as e:
            # Fail immediately on non-retryable status.
            status = getattr(e.response, "status_code", None)
            if status not in (429, 500, 502, 503, 504) or attempt == max_retries:
                # Keep detailed error text.
                try:
                    detail = e.response.text
                except Exception:
                    detail = str(e)
                raise RuntimeError(f"TTS request failed (status={status}): {detail}") from e

        # Exponential backoff.
        sleep_s = min(2 ** (attempt - 1), 16)
        time.sleep(sleep_s)

    # Unreachable by design.
    raise RuntimeError("All TTS requests failed due to unknown errors.")


def process_csv(
    pilot_csv_path: Path,
    voice_info_path: Path,
    out_dir: Path = Path("outputs/switched_stimulus/target_raw"),
) -> None:
    token = os.environ.get("TYPECAST_API_TOKEN")
    if not token:
        raise EnvironmentError("TYPECAST_API_TOKEN is not set.")

    voice_map = read_voice_map(voice_info_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    with pilot_csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required_cols = {
            "day",
            "block",
            "stimulus_num",
            "name",
            "utterance",
        }
        if not required_cols.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"Missing required columns in input CSV. Required: {required_cols}, actual: {reader.fieldnames}"
            )

        total, success, skipped = 0, 0, 0
        for row in reader:
            total += 1
            day = (row.get("day") or "").strip()
            block = (row.get("block") or "").strip()
            name = (row.get("name") or "").strip()
            text = (row.get("utterance") or "").strip()
            stim = (row.get("stimulus_num") or "").strip()

            if not day or not block or not name or not text or not stim:
                print(f"[SKIP] Missing required values: day={day!r}, block={block!r}, stimulus_num={stim!r}, name={name!r}, utterance(len)={len(text)}")
                skipped += 1
                continue

            voice_id: Optional[str] = voice_map.get(name)
            if not voice_id:
                print(f"[SKIP] No mapping for name='{name}' in voice_info CSV.")
                skipped += 1
                continue

            fname = f"day{day}block{block}_stimulus{stim}.wav"
            outpath = out_dir / fname

            try:
                audio = fetch_audio_bytes_from_tts(token=token, voice_id=voice_id, text=text)
                with outpath.open("wb") as wf:
                    wf.write(audio)
                success += 1
                print(f"[OK] {outpath}")
            except Exception as e:
                print(f"[FAIL] stimulus_num={stim}, name={name} -> {e}")

    print(f"\nDone: total={total}, success={success}, skipped={skipped}, output={out_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate switched speech WAV files from CSV using Typecast"
    )
    parser.add_argument("pilot_csv", help="Path to switched CSV")
    parser.add_argument(
        "--voice-info",
        default="data/generated/voice_info.csv",
        help="Path to voice_info CSV (default: data/generated/voice_info.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/switched_stimulus/target_raw",
        help="Output folder for generated WAV files",
    )
    args = parser.parse_args()

    pilot_csv_path = Path(args.pilot_csv)
    voice_info_path = Path(args.voice_info)
    output_dir = Path(args.output_dir)

    if not pilot_csv_path.exists():
        print(f"Input CSV not found: {pilot_csv_path}", file=sys.stderr)
        sys.exit(1)
    if not voice_info_path.exists():
        print(f"voice_info CSV not found: {voice_info_path}", file=sys.stderr)
        sys.exit(1)

    process_csv(pilot_csv_path, voice_info_path, output_dir)


if __name__ == "__main__":
    main()
