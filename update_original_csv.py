#!/usr/bin/env python3
"""
Overwrite utterance values in original CSV from regenerated CSV.
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path
from datetime import datetime


def update_csv_utterances(original_csv_path: Path, regenerated_csv_path: Path, backup: bool = True) -> None:
    """
    Overwrite utterances in original_csv_path using regenerated_csv_path.
    """
    if not original_csv_path.exists():
        print(f"[ERROR] Original CSV not found: {original_csv_path}", file=sys.stderr)
        sys.exit(1)
    
    if not regenerated_csv_path.exists():
        print(f"[ERROR] Regenerated CSV not found: {regenerated_csv_path}", file=sys.stderr)
        sys.exit(1)
    
    # Create backup.
    if backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = original_csv_path.with_suffix(f".backup_{timestamp}.csv")
        shutil.copy2(original_csv_path, backup_path)
        print(f"[INFO] Backup created: {backup_path}")
    
    # Read utterance values from regenerated CSV.
    regenerated_map = {}  # (day, block, stimulus_num) -> utterance
    with regenerated_csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            day = row.get("day", "").strip()
            block = row.get("block", "").strip()
            stimulus_num = row.get("stimulus_num", "").strip()
            utterance = row.get("utterance", "").strip()
            
            if day and block and stimulus_num and utterance:
                key = (day, block, stimulus_num)
                regenerated_map[key] = utterance
    
    print(f"[INFO] Loaded {len(regenerated_map)} regenerated utterances")
    
    # Read original CSV and update rows.
    updated_rows = []
    updated_count = 0
    not_found_count = 0
    
    with original_csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            day = row.get("day", "").strip()
            block = row.get("block", "").strip()
            stimulus_num = row.get("stimulus_num", "").strip()
            
            if day and block and stimulus_num:
                key = (day, block, stimulus_num)
                if key in regenerated_map:
                    # Update utterance.
                    row["utterance"] = regenerated_map[key]
                    updated_count += 1
                    print(f"[UPDATE] day={day}, block={block}, stimulus_num={stimulus_num}")
                else:
                    # Keep original when key is not in regenerated CSV.
                    not_found_count += 1
            else:
                # Keep original when required fields are missing.
                pass
            
            updated_rows.append(row)
    
    # Save updated CSV.
    with original_csv_path.open("w", newline="", encoding="utf-8") as f:
        if fieldnames:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)
    
    print(f"\n[COMPLETE] Update finished:")
    print(f"  - Updated rows: {updated_count}")
    print(f"  - Rows missing in regenerated CSV: {not_found_count}")
    print(f"  - Total rows: {len(updated_rows)}")
    print(f"  - Original file: {original_csv_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Overwrite original utterances from regenerated CSV"
    )
    parser.add_argument(
        "split",
        help="Split name (e.g., target, interference)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable backup creation",
    )
    args = parser.parse_args()
    
    original_csv_path = Path(f"data/generated/switched/text_speaker_switched_{args.split}.csv")
    regenerated_csv_path = Path(f"data/generated/switched/text_speaker_switched_{args.split}_regenerated.csv")
    
    update_csv_utterances(
        original_csv_path=original_csv_path,
        regenerated_csv_path=regenerated_csv_path,
        backup=not args.no_backup
    )


if __name__ == "__main__":
    main()
