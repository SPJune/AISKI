#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI


OUTLIER_REQUIRED_COLS = {"day", "block", "stimulus_num", "duration_sec"}
SOURCE_REQUIRED_COLS = {"stimulus_num", "utterance"}


def word_count(text: str) -> int:
    tokens = re.findall(r"\S+", text or "")
    return len(tokens)


def predict_duration_sec(reference_duration: float, reference_words: int, candidate_words: int) -> float:
    safe_ref_words = max(1, reference_words)
    safe_candidate_words = max(1, candidate_words)
    return reference_duration * (safe_candidate_words / safe_ref_words)


def compute_target_word_plan(current_words: int, current_duration: float, target_duration: float) -> Dict[str, int]:
    raw_target = max(1, int(round(current_words * target_duration / current_duration)))

    # Conservative bounds:
    # - When text is short (duration < target), allow only moderate expansion.
    # - When text is long (duration > target), allow stronger reduction.
    if current_duration < target_duration:
        capped_target = min(raw_target, int(math.ceil(current_words * 1.20)), current_words + 20)
    else:
        capped_target = max(raw_target, int(math.floor(current_words * 0.75)), current_words - 35)

    target_words = max(1, capped_target)
    lower_bound = max(1, target_words - 4)
    upper_bound = max(lower_bound, target_words + 4)

    return {
        "raw_target_words": raw_target,
        "target_words": target_words,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    }


def parse_outliers(outlier_csv: Path) -> Dict[Tuple[int, int], List[Dict[str, str]]]:
    grouped: Dict[Tuple[int, int], List[Dict[str, str]]] = defaultdict(list)
    with outlier_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not OUTLIER_REQUIRED_COLS.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"Missing required columns in outlier CSV. "
                f"Required: {OUTLIER_REQUIRED_COLS}, actual: {reader.fieldnames}"
            )
        for row in reader:
            try:
                day = int((row.get("day") or "").strip())
                block = int((row.get("block") or "").strip())
                stim = int((row.get("stimulus_num") or "").strip())
                duration = float((row.get("duration_sec") or "").strip())
            except ValueError:
                continue
            grouped[(day, block)].append(
                {
                    "stimulus_num": str(stim),
                    "duration_sec": f"{duration}",
                }
            )
    return grouped


def build_prompt(
    original_text: str,
    current_words: int,
    current_duration: float,
    target_duration: float,
    plan: Dict[str, int],
) -> Tuple[str, str]:
    words_to_reduce = current_words - plan["target_words"]
    words_to_increase = max(0, -words_to_reduce)

    system_prompt = (
        "You rewrite English spoken-style stimulus text.\n"
        "Requirements:\n"
        "- Keep the same core meaning, tone, and narrative flow.\n"
        "- Keep natural spoken English.\n"
        "- Avoid changing key entities, events, and causal relations.\n"
        "- Do not add new major events, locations, or characters.\n"
        "- Return plain text only, no JSON and no extra commentary."
    )

    # User requested: include how many words to reduce for 30s in prompt.
    # This field is always included as words_to_reduce_for_30s.
    user_prompt = (
        f"Target duration: {target_duration:.1f} seconds\n"
        f"Current duration: {current_duration:.4f} seconds\n"
        f"Current word count: {current_words}\n"
        f"Estimated target word count for {target_duration:.1f}s (raw): {plan['raw_target_words']}\n"
        f"Conservative target word count: {plan['target_words']}\n"
        f"Allowed final word-count range: {plan['lower_bound']} to {plan['upper_bound']}\n"
        f"words_to_reduce_for_30s: {words_to_reduce}\n"
        f"words_to_increase_if_negative: {words_to_increase}\n\n"
        "Rewrite the text accordingly while preserving meaning.\n"
        "Original text:\n"
        f"{original_text}"
    )
    return system_prompt, user_prompt


def regenerate_one_text(
    client: OpenAI,
    model: str,
    original_text: str,
    current_duration: float,
    target_duration: float,
) -> str:
    current_words = word_count(original_text)
    plan = compute_target_word_plan(
        current_words=current_words,
        current_duration=current_duration,
        target_duration=target_duration,
    )
    system_prompt, user_prompt = build_prompt(
        original_text=original_text,
        current_words=current_words,
        current_duration=current_duration,
        target_duration=target_duration,
        plan=plan,
    )
    def _call(prompt_system: str, prompt_user: str, temperature: float) -> str:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user},
            ],
            temperature=temperature,
            top_p=0.9,
            max_output_tokens=1200,
        )
        out = (resp.output_text or "").strip()
        if out.startswith("```"):
            out = re.sub(r"^```(?:text|markdown)?\s*|\s*```$", "", out, flags=re.S).strip()
        return out

    first = _call(system_prompt, user_prompt, temperature=0.5)
    first_words = word_count(first)
    if plan["lower_bound"] <= first_words <= plan["upper_bound"]:
        return first

    # Repair pass: aggressively constrain the output length.
    repair_prompt = (
        "The previous rewrite is outside the allowed word-count range.\n"
        f"Current rewritten word count: {first_words}\n"
        f"Required range: {plan['lower_bound']} to {plan['upper_bound']}\n"
        "Rewrite again with minimal edits while keeping meaning identical.\n"
        "Return plain text only."
    )
    second = _call(system_prompt, f"{user_prompt}\n\n{repair_prompt}", temperature=0.2)
    return second


def repair_text_for_predicted_duration(
    client: OpenAI,
    model: str,
    text_to_repair: str,
    reference_duration: float,
    reference_words: int,
    min_sec: float,
    max_sec: float,
    target_duration: float,
) -> str:
    predicted_now = predict_duration_sec(
        reference_duration=reference_duration,
        reference_words=reference_words,
        candidate_words=word_count(text_to_repair),
    )
    target_words = max(1, int(round(reference_words * target_duration / reference_duration)))
    lower_words = max(1, int(round(reference_words * min_sec / reference_duration)))
    upper_words = max(lower_words, int(round(reference_words * max_sec / reference_duration)))

    system_prompt = (
        "You rewrite English spoken-style stimulus text.\n"
        "Keep the same meaning, entities, and event flow.\n"
        "Do not add new major events or characters.\n"
        "Return plain text only."
    )
    user_prompt = (
        f"Predicted duration from word-based model: {predicted_now:.4f}s\n"
        f"Required duration range: {min_sec:.1f}s to {max_sec:.1f}s\n"
        f"Reference words: {reference_words}\n"
        f"Target words for {target_duration:.1f}s: {target_words}\n"
        f"Allowed word-count range: {lower_words} to {upper_words}\n\n"
        "Rewrite the text with minimal edits so the estimated duration falls inside the required range.\n"
        "Current text:\n"
        f"{text_to_repair}"
    )
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        top_p=0.9,
        max_output_tokens=1200,
    )
    out = (resp.output_text or "").strip()
    if out.startswith("```"):
        out = re.sub(r"^```(?:text|markdown)?\s*|\s*```$", "", out, flags=re.S).strip()
    return out


def backup_file(path: Path, backup_root: Path) -> Path:
    backup_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_root / f"{path.stem}.backup_{ts}{path.suffix}"
    shutil.copy2(path, backup_path)
    return backup_path


def process_one_block_file(
    csv_path: Path,
    outlier_rows: List[Dict[str, str]],
    client: OpenAI,
    model: str,
    target_duration: float,
    min_sec: float,
    max_sec: float,
    dry_run: bool,
    sleep_sec: float,
) -> Tuple[int, int]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not SOURCE_REQUIRED_COLS.issubset(set(fieldnames)):
            raise ValueError(
                f"Missing required columns in source CSV {csv_path}. "
                f"Required: {SOURCE_REQUIRED_COLS}, actual: {fieldnames}"
            )
        rows = list(reader)

    outlier_map = {int(x["stimulus_num"]): float(x["duration_sec"]) for x in outlier_rows}

    changed = 0
    attempted = 0
    for row in rows:
        stim_raw = (row.get("stimulus_num") or "").strip()
        try:
            stim = int(stim_raw)
        except ValueError:
            continue
        if stim not in outlier_map:
            continue

        original_text = (row.get("utterance") or "").strip()
        if not original_text:
            print(f"[SKIP] Empty utterance in {csv_path.name} stimulus={stim}")
            continue

        attempted += 1
        duration_sec = outlier_map[stim]
        try:
            original_words = word_count(original_text)
            new_text = regenerate_one_text(
                client=client,
                model=model,
                original_text=original_text,
                current_duration=duration_sec,
                target_duration=target_duration,
            )
            if new_text:
                predicted_sec = predict_duration_sec(
                    reference_duration=duration_sec,
                    reference_words=original_words,
                    candidate_words=word_count(new_text),
                )

                # One additional adjustment pass if predicted duration is still out of range.
                if predicted_sec < min_sec or predicted_sec > max_sec:
                    repaired_text = repair_text_for_predicted_duration(
                        client=client,
                        model=model,
                        text_to_repair=new_text,
                        reference_duration=duration_sec,
                        reference_words=original_words,
                        min_sec=min_sec,
                        max_sec=max_sec,
                        target_duration=target_duration,
                    )
                    if repaired_text:
                        repaired_predicted_sec = predict_duration_sec(
                            reference_duration=duration_sec,
                            reference_words=original_words,
                            candidate_words=word_count(repaired_text),
                        )
                        # Prefer the repaired version when it is closer to target duration.
                        if abs(repaired_predicted_sec - target_duration) <= abs(predicted_sec - target_duration):
                            new_text = repaired_text
                            predicted_sec = repaired_predicted_sec

                row["utterance"] = new_text
                changed += 1
                print(
                    f"[OK] {csv_path.name} stimulus={stim} "
                    f"(words {word_count(original_text)} -> {word_count(new_text)}, "
                    f"ref_duration={duration_sec:.2f}s, predicted={predicted_sec:.2f}s)"
                )
            else:
                print(f"[SKIP] Empty model output: {csv_path.name} stimulus={stim}")
        except Exception as e:
            print(f"[FAIL] {csv_path.name} stimulus={stim} -> {e}")
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    if changed > 0 and not dry_run:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return attempted, changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate outlier utterances in data/reference_eng using OpenAI."
    )
    parser.add_argument(
        "--outlier-csv",
        default="data/generated/stimulus_eng_out_of_range.csv",
        help="Outlier CSV produced by generate_speech_eng.py",
    )
    parser.add_argument(
        "--reference-dir",
        default="data/reference_eng",
        help="Directory containing day{i}block{j}.csv",
    )
    parser.add_argument(
        "--target-duration",
        type=float,
        default=30.0,
        help="Target duration in seconds",
    )
    parser.add_argument(
        "--min-sec",
        type=float,
        default=25.0,
        help="Minimum acceptable predicted duration",
    )
    parser.add_argument(
        "--max-sec",
        type=float,
        default=35.0,
        help="Maximum acceptable predicted duration",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4.1"),
        help="OpenAI model name",
    )
    parser.add_argument(
        "--backup-dir",
        default="data/reference_eng/_backup",
        help="Backup directory before in-place update",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Sleep seconds between OpenAI calls",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run generation without writing updated CSV files",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    outlier_csv = Path(args.outlier_csv)
    reference_dir = Path(args.reference_dir)
    backup_dir = Path(args.backup_dir)

    if not outlier_csv.exists():
        print(f"ERROR: outlier CSV not found: {outlier_csv}", file=sys.stderr)
        sys.exit(1)
    if not reference_dir.exists():
        print(f"ERROR: reference directory not found: {reference_dir}", file=sys.stderr)
        sys.exit(1)

    grouped = parse_outliers(outlier_csv)
    if not grouped:
        print("No outlier rows found. Nothing to regenerate.")
        return

    client = OpenAI(api_key=api_key)

    total_attempted = 0
    total_changed = 0
    total_files = 0

    for (day, block), rows in sorted(grouped.items()):
        csv_path = reference_dir / f"day{day}block{block}.csv"
        if not csv_path.exists():
            print(f"[WARN] Missing source CSV: {csv_path}")
            continue

        total_files += 1
        print(f"\n=== Processing {csv_path.name} ===")
        if not args.dry_run:
            backup_path = backup_file(csv_path, backup_dir)
            print(f"[INFO] Backup: {backup_path}")

        attempted, changed = process_one_block_file(
            csv_path=csv_path,
            outlier_rows=rows,
            client=client,
            model=args.model,
            target_duration=args.target_duration,
            min_sec=args.min_sec,
            max_sec=args.max_sec,
            dry_run=args.dry_run,
            sleep_sec=args.sleep,
        )
        total_attempted += attempted
        total_changed += changed

    print("\n=== Done ===")
    print(f"Processed files: {total_files}")
    print(f"Attempted regenerations: {total_attempted}")
    print(f"Updated utterances: {total_changed}")
    if args.dry_run:
        print("Dry-run mode: no files were written.")


if __name__ == "__main__":
    main()
