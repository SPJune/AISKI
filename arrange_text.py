import pandas as pd
import sys

def sort_and_add_num(csv_path: str):
    # Load CSV.
    df = pd.read_csv(csv_path)
    
    # Sort ascending by length_no_space.
    df = df.sort_values(by="length_no_space", ascending=True).reset_index(drop=True)
    
    # Insert num column at front (starting from 1).
    df.insert(0, "num", range(1, len(df) + 1))
    
    # Overwrite original file.
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <csv_path>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    sort_and_add_num(csv_path)
    print(f"Sorted and inserted num column: {csv_path}")
