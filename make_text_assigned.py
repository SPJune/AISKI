#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path

# --------- Input file paths ----------
pairs_path = Path("data/generated/pairs.csv")
text_path = Path("data/generated/text.csv")
voice_path = Path("data/generated/voice_info.csv")
out_path = Path("data/generated/text_aligned.csv")
# ------------------------------------

def load_inputs():
    df_pairs = pd.read_csv(pairs_path)
    df_text  = pd.read_csv(text_path)
    df_voice = pd.read_csv(voice_path)

    # Normalize numeric columns.
    df_text["length_no_space"] = pd.to_numeric(df_text["length_no_space"], errors="coerce")
    df_voice["C"] = pd.to_numeric(df_voice["C"], errors="coerce")

    return df_pairs, df_text, df_voice

def build_target_C(df_voice: pd.DataFrame):
    return dict(zip(df_voice["name"], df_voice["C"].astype(int)))

def pick_row_strict_unused(df_text_available: pd.DataFrame, target_C: int):
    if df_text_available.empty:
        return None
    df = df_text_available.copy()
    df["len"] = df["length_no_space"].astype(int)
    df["is_ge"] = (df["len"] >= target_C).astype(int)
    df["diff"] = (df["len"] - target_C).abs()
    df = df.sort_values(by=["is_ge","diff","len"], ascending=[False, True, False])
    return df.index[0]

def assign_for_pair_strict(stimulus_num, name, targetC_map, df_text_all, used_idx_global, forbidden_main_topics):
    C_target = int(targetC_map[name])

    mask_unused = ~df_text_all.index.isin(used_idx_global)
    mask_allowed_topic = ~df_text_all["main_topic"].isin(forbidden_main_topics) if forbidden_main_topics else True
    candidates = df_text_all[mask_unused & mask_allowed_topic]

    idx = pick_row_strict_unused(candidates, C_target)
    if idx is None:
        raise RuntimeError(f"[stimulus {stimulus_num}] Not enough candidate utterances for name='{name}'.")

    row = df_text_all.loc[idx]
    used_idx_global.add(idx)

    out_row = {
        "stimulus_num": stimulus_num,
        "name": name,
        "main_topic": row["main_topic"],
        "sub_topic": row["sub_topic"],
        "C": row["C"],
        "utterance": row["utterance"],
        "question": row["question"],
        "option1": row["option1"],
        "option2": row["option2"],
        "option3": row["option3"],
        "option4": row["option4"],
        "answer": row["answer"],
        "answer_index": row["answer_index"],
        "length_no_space": row["length_no_space"],
        "target_length": C_target,  # Added to track per-speaker target length.
    }
    return out_row

def main():
    df_pairs, df_text, df_voice = load_inputs()
    targetC_map = build_target_C(df_voice)

    df_text = df_text.reset_index(drop=True)
    used_idx_global = set()
    outputs = []

    for i, pair in df_pairs.reset_index(drop=True).iterrows():
        stimulus_num = i + 1
        name1, name2 = pair["name1"], pair["name2"]

        row1 = assign_for_pair_strict(stimulus_num, name1, targetC_map, df_text, used_idx_global, set())
        row2 = assign_for_pair_strict(stimulus_num, name2, targetC_map, df_text, used_idx_global, {row1["main_topic"]})

        outputs.extend([row1, row2])

    df_out = pd.DataFrame(outputs, columns=[
        "stimulus_num","name","main_topic","sub_topic","C","utterance",
        "question","option1","option2","option3","option4","answer","answer_index",
        "length_no_space","target_length"
    ])

    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path} ({len(df_out)} rows; expected {len(df_pairs)*2})")

if __name__ == "__main__":
    main()
