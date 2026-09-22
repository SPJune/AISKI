#!/usr/bin/env python3
"""Regenerate switched speech when durations are out of tolerance."""
import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import soundfile as sf
from openai import OpenAI

# OpenAI API configuration
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise EnvironmentError("Set OPENAI_API_KEY environment variable.")
openai_client = OpenAI(api_key=API_KEY)

# Typecast TTS configuration
TYPECAST_TTS_URL = "https://api.typecast.ai/v1/text-to-speech"
TYPECAST_API_TOKEN = os.environ.get("TYPECAST_API_TOKEN")
if not TYPECAST_API_TOKEN:
    raise EnvironmentError("Set TYPECAST_API_TOKEN environment variable.")


def H(key: str) -> Dict[str, str]:
    return {
        "X-API-KEY": key,
        "Accept": "application/json, audio/wav",
        "Content-Type": "application/json",
    }


def get_audio_duration(wav_path: Path) -> float:
    """Return WAV duration in seconds."""
    try:
        data, sr = sf.read(wav_path)
        duration = len(data) / sr
        return duration
    except Exception as e:
        print(f"[ERROR] Failed to read {wav_path}: {e}", file=sys.stderr)
        return 0.0


def parse_wav_filename(filename: str) -> Optional[Tuple[str, str, str]]:
    """Extract day, block, stimulus_num from day{day}block{block}_stimulus{n}.wav."""
    match = re.match(r"day(\d+)block(\d+)_stimulus(\d+)\.wav", filename)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None


def find_csv_row(csv_path: Path, day: str, block: str, stimulus_num: str) -> Optional[Dict]:
    """Find matching CSV row by day, block, and stimulus_num."""
    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("day", "").strip() == day and
                row.get("block", "").strip() == block and
                row.get("stimulus_num", "").strip() == stimulus_num):
                return row
    return None


def adjust_text_length(original_text: str, current_duration: float, target_duration: float) -> str:
    """
    Adjust text length with OpenAI while preserving context and tone.
    Uses a softened ratio to avoid extreme length changes.
    """
    ratio = target_duration / current_duration
    # Use softened ratio so edits remain conservative.
    adjusted_ratio = math.pow(ratio, 0.025)
    
    # Determine whether text should be expanded or shortened.
    if adjusted_ratio > 1.0:
        direction = "늘려서"
        target_chars = int(len(original_text) * adjusted_ratio)
    else:
        direction = "줄여서"
        target_chars = int(len(original_text) * adjusted_ratio)
    
    system_prompt = f"""당신은 한국어 텍스트의 길이를 조정하는 어시스턴트입니다.
요구사항:
- 주어진 텍스트의 맥락, 의미, 톤을 완전히 유지하세요.
- 텍스트를 아주 조금 {direction} 약 {target_chars}자 정도로 조정하세요.
- 문장의 핵심 내용은 그대로 유지하되, 세부 설명을 추가하거나 간결하게 표현하세요.
- 출력은 수정된 텍스트만 반환하세요. JSON이나 다른 형식 없이 순수 텍스트만."""

    user_prompt = f"""다음 텍스트를 {direction} 약 {target_chars}자 정도로 조정해주세요. 맥락과 의미는 그대로 유지하세요.

{original_text}"""

    try:
        resp = openai_client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            top_p=0.9,
            max_output_tokens=800,
        )
        content = resp.output_text.strip()
        # Remove markdown code fences if present.
        if content.startswith("```"):
            content = re.sub(r"^```(?:json|text)?\s*|\s*```$", "", content, flags=re.S)
        # Try JSON fallback parsing.
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "utterance" in data:
                return data["utterance"].strip()
            elif isinstance(data, dict) and "text" in data:
                return data["text"].strip()
        except json.JSONDecodeError:
            pass
        return content.strip()
    except Exception as e:
        print(f"[ERROR] Failed to adjust text: {e}", file=sys.stderr)
        return original_text


def fetch_audio_bytes_from_tts(
    token: str,
    voice_id: str,
    text: str,
    timeout: float = 60.0,
    max_retries: int = 5,
) -> bytes:
    """Generate speech with Typecast TTS."""
    headers = H(token)
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
    }

    def _post_tts():
        return requests.post(TYPECAST_TTS_URL, headers=headers, json=payload, timeout=timeout)

    for attempt in range(1, max_retries + 1):
        try:
            resp = _post_tts()
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"Retryable status {resp.status_code}", response=resp)

            resp.raise_for_status()
            ctype = resp.headers.get("Content-Type", "").lower()

            if "audio" in ctype or "octet-stream" in ctype:
                return resp.content

            if "application/json" in ctype or ctype.endswith("+json"):
                data = resp.json()
                candidate_keys = [
                    "audio_url", "url", "result_url", "audio",
                    "audioPath", "file_url", "wav_url",
                ]
                for k in candidate_keys:
                    if k in data and isinstance(data[k], str) and data[k].startswith(("http://", "https://")):
                        audio_url = data[k]
                        a = requests.get(audio_url, timeout=timeout)
                        a.raise_for_status()
                        return a.content

                if "audio_base64" in data and isinstance(data["audio_base64"], str):
                    import base64
                    return base64.b64decode(data["audio_base64"])

                raise RuntimeError(f"Could not find audio URL/data in TTS JSON response. keys={list(data.keys())}")

            raise RuntimeError(f"Unexpected Content-Type: {ctype}")

        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt == max_retries:
                raise
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status not in (429, 500, 502, 503, 504) or attempt == max_retries:
                try:
                    detail = e.response.text
                except Exception:
                    detail = str(e)
                raise RuntimeError(f"TTS request failed (status={status}): {detail}") from e

        sleep_s = min(2 ** (attempt - 1), 16)
        time.sleep(sleep_s)

    raise RuntimeError("All TTS requests failed due to unknown errors.")


def read_voice_map(voice_info_path: Path) -> Dict[str, str]:
    """Load name -> voice_id mapping from voice_info CSV."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Check switched WAV durations and regenerate out-of-range files"
    )
    parser.add_argument("split", help="Split name (e.g., target, interference)")
    parser.add_argument(
        "--voice-info",
        default="data/generated/voice_info.csv",
        help="Path to voice_info CSV",
    )
    parser.add_argument(
        "--target-duration",
        type=float,
        default=30.0,
        help="Target audio duration in seconds",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=6.0,
        help="Tolerance in seconds",
    )
    args = parser.parse_args()

    # Path setup
    wav_dir = Path(f"outputs/switched_stimulus/{args.split}_raw")
    csv_path = Path(f"data/generated/switched/text_speaker_switched_{args.split}.csv")
    output_dir = Path(f"outputs/switched_stimulus/{args.split}_regenerate")
    voice_info_path = Path(args.voice_info)
    output_csv_path = Path(f"data/generated/switched/text_speaker_switched_{args.split}_regenerated.csv")

    # Validate paths
    if not wav_dir.exists():
        print(f"[ERROR] WAV directory not found: {wav_dir}", file=sys.stderr)
        sys.exit(1)
    if not csv_path.exists():
        print(f"[ERROR] CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    if not voice_info_path.exists():
        print(f"[ERROR] voice_info CSV not found: {voice_info_path}", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load voice map
    voice_map = read_voice_map(voice_info_path)

    # Read CSV
    rows_to_modify = []
    rows_to_keep = []
    modified_rows = []

    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        all_rows = list(reader)

    # Check WAV files
    wav_files = list(wav_dir.glob("*.wav"))
    print(f"[INFO] Found {len(wav_files)} WAV files")

    target_duration = args.target_duration
    tolerance = args.tolerance
    min_duration = target_duration - tolerance
    max_duration = target_duration + tolerance

    files_to_regenerate = []
    files_to_copy = []

    for wav_file in wav_files:
        duration = get_audio_duration(wav_file)
        print(f"[INFO] {wav_file.name}: {duration:.2f}s")

        parsed = parse_wav_filename(wav_file.name)
        if not parsed:
            print(f"[WARN] Invalid filename format: {wav_file.name}")
            continue

        day, block, stimulus_num = parsed
        csv_row = find_csv_row(csv_path, day, block, stimulus_num)

        if not csv_row:
            print(f"[WARN] CSV row not found: day={day}, block={block}, stimulus_num={stimulus_num}")
            continue

        if duration < min_duration or duration > max_duration:
            print(f"[MODIFY] {wav_file.name}: {duration:.2f}s -> regeneration required")
            files_to_regenerate.append((wav_file, duration, csv_row))
        else:
            print(f"[OK] {wav_file.name}: {duration:.2f}s -> copy")
            files_to_copy.append((wav_file, duration, csv_row))

    # Process files
    print(f"\n[INFO] Files to regenerate: {len(files_to_regenerate)}")
    print(f"[INFO] Files to copy: {len(files_to_copy)}")

    # Build regenerated CSV rows
    for wav_file, duration, csv_row in files_to_regenerate:
        original_text = csv_row.get("utterance", "").strip()
        if not original_text:
            print(f"[WARN] Empty utterance: {wav_file.name}")
            modified_rows.append(csv_row)
            continue

        print(f"\n[ADJUST] {wav_file.name}: adjusting text...")
        adjusted_text = adjust_text_length(original_text, duration, target_duration)
        print(f"[ADJUST] Original chars: {len(original_text)}, adjusted chars: {len(adjusted_text)}")

        # Copy row and replace utterance
        new_row = csv_row.copy()
        new_row["utterance"] = adjusted_text
        modified_rows.append(new_row)

    # Append rows that do not need changes
    for _, _, csv_row in files_to_copy:
        modified_rows.append(csv_row)

    # Save regenerated CSV
    with output_csv_path.open("w", newline="", encoding="utf-8") as f:
        if fieldnames:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in modified_rows:
                writer.writerow(row)

    print(f"\n[INFO] Saved regenerated CSV: {output_csv_path}")

    # Regenerate audio using updated CSV
    print(f"\n[INFO] Starting audio regeneration...")
    success_count = 0
    fail_count = 0

    # Load regenerated CSV
    regenerate_map = {}  # (day, block, stimulus_num) -> row
    with output_csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            day = row.get("day", "").strip()
            block = row.get("block", "").strip()
            stimulus_num = row.get("stimulus_num", "").strip()
            if day and block and stimulus_num:
                regenerate_map[(day, block, stimulus_num)] = row

    for wav_file, duration, csv_row in files_to_regenerate:
        day, block, stimulus_num = parse_wav_filename(wav_file.name)
        name = csv_row.get("name", "").strip()
        
        # Find row in regenerated CSV
        updated_row = regenerate_map.get((day, block, stimulus_num))
        if not updated_row:
            print(f"[WARN] Row not found in regenerated CSV: {wav_file.name}")
            fail_count += 1
            continue

        adjusted_text = updated_row.get("utterance", "").strip()
        if not adjusted_text:
            print(f"[WARN] Empty utterance: {wav_file.name}")
            fail_count += 1
            continue

        voice_id = voice_map.get(name)
        if not voice_id:
            print(f"[WARN] Missing voice_id for name={name}")
            fail_count += 1
            continue

        output_path = output_dir / wav_file.name
        try:
            print(f"[GENERATE] Generating {wav_file.name}...")
            audio = fetch_audio_bytes_from_tts(
                token=TYPECAST_API_TOKEN,
                voice_id=voice_id,
                text=adjusted_text
            )
            with output_path.open("wb") as wf:
                wf.write(audio)
            
            # Verify regenerated duration
            new_duration = get_audio_duration(output_path)
            print(f"[OK] {wav_file.name}: {new_duration:.2f}s (original: {duration:.2f}s)")
            success_count += 1
        except Exception as e:
            print(f"[FAIL] {wav_file.name}: {e}")
            fail_count += 1

    # Copy unchanged files
    print(f"\n[INFO] Starting file copy...")
    for wav_file, duration, _ in files_to_copy:
        output_path = output_dir / wav_file.name
        try:
            shutil.copy2(wav_file, output_path)
            print(f"[COPY] {wav_file.name}: {duration:.2f}s")
        except Exception as e:
            print(f"[FAIL] Failed to copy {wav_file.name}: {e}")

    # Print final duration summary
    print(f"\n[INFO] Final audio duration check...")
    final_wav_files = list(output_dir.glob("*.wav"))
    for wav_file in sorted(final_wav_files):
        duration = get_audio_duration(wav_file)
        print(f"[FINAL] {wav_file.name}: {duration:.2f}s")

    print(f"\n[COMPLETE] Regenerated: success={success_count}, fail={fail_count}")
    print(f"[COMPLETE] Copied: {len(files_to_copy)}")
    print(f"[COMPLETE] Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
