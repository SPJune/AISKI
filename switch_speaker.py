import os
import pandas as pd

# CSV paths
text_aligned_path = "data/generated/text_aligned.csv"
mixture_infos_split_dir = "data/generated/mixture_infos_split"

# Output paths
output_target_path = "data/generated/switched/text_speaker_switched_target.csv"
output_interference_path = "data/generated/switched/text_speaker_switched_interference.csv"

print("Loading text_aligned.csv...")
text_aligned_df = pd.read_csv(text_aligned_path)
os.makedirs("data/generated/switched", exist_ok=True)

target_rows = []
interference_rows = []

# Process day{1..3}block{1..3}
for i in [1, 2, 3]:
    for j in [1, 2, 3]:
        info_path = os.path.join(mixture_infos_split_dir, f"day{i}block{j}_info.csv")
        switched_path = os.path.join(mixture_infos_split_dir, f"day{i}block{j}_info_switched.csv")

        print(f"\nProcessing: day{i}block{j}")
        info_df = pd.read_csv(info_path)
        switched_df = info_df.copy()

        # Keep first 15 rows unchanged. For rows 16..30, switch target/interference
        # based on the corresponding pair in rows 1..15.
        for k in range(1, 16):
            row_k_idx = k + 14
            row_k15_idx = k - 1
            switched_df.loc[row_k_idx, "name_target"] = switched_df.loc[row_k15_idx, "name_interference"]
            switched_df.loc[row_k_idx, "name_interference"] = switched_df.loc[row_k15_idx, "name_target"]

        switched_df.to_csv(switched_path, index=False)
        print(f"  Saved: {switched_path}")

        # Build switched target/interference rows using old_num lookup.
        for k in range(1, 16):
            row_idx = k + 14
            old_num = info_df.loc[row_idx, "old_num"]
            stimulus_num = info_df.loc[row_idx, "stimulus_num"]
            name_target_switched = switched_df.loc[row_idx, "name_target"]
            name_interference_switched = switched_df.loc[row_idx, "name_interference"]
            matching_rows = text_aligned_df[text_aligned_df["stimulus_num"] == old_num]

            if len(matching_rows) == 2:
                for _, matched_row in matching_rows.iterrows():
                    matched_name = matched_row["name"]
                    if matched_name == info_df.loc[row_idx, "name_target"]:
                        row_dict = matched_row.to_dict()
                        row_dict["name"] = name_target_switched
                        result_dict = {"day": i, "block": j, "stimulus_num": stimulus_num, "old_num": old_num}
                        for key, value in row_dict.items():
                            if key != "stimulus_num":
                                result_dict[key] = value
                        target_rows.append(result_dict)
                    elif matched_name == info_df.loc[row_idx, "name_interference"]:
                        row_dict = matched_row.to_dict()
                        row_dict["name"] = name_interference_switched
                        result_dict = {"day": i, "block": j, "stimulus_num": stimulus_num, "old_num": old_num}
                        for key, value in row_dict.items():
                            if key != "stimulus_num":
                                result_dict[key] = value
                        interference_rows.append(result_dict)
            else:
                print(f"  Warning: old_num={old_num} matched {len(matching_rows)} rows (expected 2).")

if target_rows:
    target_df = pd.DataFrame(target_rows)
    base_cols = [
        "stimulus_num", "name", "main_topic", "sub_topic", "C", "utterance", "question",
        "option1", "option2", "option3", "option4", "answer", "answer_index",
        "length_no_space", "target_length",
    ]
    new_cols = ["day", "block", "stimulus_num", "old_num"]
    for col in base_cols:
        if col != "stimulus_num" and col in target_df.columns:
            new_cols.append(col)
    for col in target_df.columns:
        if col not in new_cols:
            new_cols.append(col)
    target_df = target_df[new_cols]
    target_df.to_csv(output_target_path, index=False)
    print(f"\nSaved: {output_target_path} ({len(target_df)} rows)")

if interference_rows:
    interference_df = pd.DataFrame(interference_rows)
    base_cols = [
        "stimulus_num", "name", "main_topic", "sub_topic", "C", "utterance", "question",
        "option1", "option2", "option3", "option4", "answer", "answer_index",
        "length_no_space", "target_length",
    ]
    new_cols = ["day", "block", "stimulus_num", "old_num"]
    for col in base_cols:
        if col != "stimulus_num" and col in interference_df.columns:
            new_cols.append(col)
    for col in interference_df.columns:
        if col not in new_cols:
            new_cols.append(col)
    interference_df = interference_df[new_cols]
    interference_df.to_csv(output_interference_path, index=False)
    print(f"Saved: {output_interference_path} ({len(interference_df)} rows)")

print("\nAll done.")
