# generate_stimuli_all.py
import os, csv, re, time, json, argparse, sys
from pathlib import Path
from typing import Dict, Any
from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

def nospace_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))

def ensure_headers(csv_path: Path):
    headers = ["main_topic","sub_topic","C","utterance","question",
               "option1","option2","option3","option4",
               "answer","answer_index","length_no_space"]
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)

def append_row(csv_path: Path, row: Dict[str, Any]):
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            row["main_topic"], row["sub_topic"], row["C"],
            row["utterance"], row["question"],
            row["options"][0], row["options"][1], row["options"][2], row["options"][3],
            row["answer"], row["answer_index"], row["length_no_space"]
        ])

SYSTEM_PROMPT = """당신은 EEG 실험용 한국어 음성 자극(발화+퀴즈)을 생성하는 어시스턴트입니다.
요구사항:
- 발화는 공백 제외 {m}자 이상, {M}자 이하로 작성하세요.
- 발화는 상식적인 지식 설명보단 개인적 경험을 설명하는 일기 형식으로 작성하세요.
- 발화의 시작이 "아침에, 오늘은, 어제는"과 같이 시작되는 것을 기피하세요.
- 퀴즈: 4지선다, 1번은 항상 '잘 모르겠음'.
- 퀴즈는 발화를 듣지 않고 풀 수 있는 상식 문제는 제외하세요.
- 이전에 생성된 발화 및 퀴즈와 유사하지 않게 새롭게 작성하세요. 같은 주제라도 새로운 사건, 새로운 시점, 새로운 인물이나 세부 상황을 사용하세요.
- 퀴즈도 발화 내용에 반드시 의존해야 하며, 같은 유형의 질문이 반복되지 않도록 변형하세요.
- 출력은 JSON 한 객체만.
스키마:
{{
  "utterance": "<공백 제외 {m}자 이상 {M}자 이하 한국어 문장>",
  "quiz": {{
    "question": "<문장>",
    "options": ["잘 모르겠음","<보기2>","<보기3>","<보기4>"],
    "answer_index": <정답 인덱스(2~4)>
  }}
}}
"""

USER_PROMPT = """대주제: {main_topic}
소주제: {sub_topic}
조건: 공백 제외 {C}자 발화 1개와 퀴즈를 위 스키마로 생성."""

def call_model(main_topic: str, sub_topic: str, C: int) -> Dict[str, Any]:
    sys_prompt = SYSTEM_PROMPT.format(m=C, M=C+20)
    user_prompt = USER_PROMPT.format(main_topic=main_topic, sub_topic=sub_topic, C=C)
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.8,
        top_p=0.9,
        max_output_tokens=800,
    )
    content = resp.output_text.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.S)
    data = json.loads(content)
    return {
        "utterance": data["utterance"].strip(),
        "question": data["quiz"]["question"].strip(),
        "options": [opt.strip() for opt in data["quiz"]["options"]],
        "answer_index": int(data["quiz"]["answer_index"])
    }

def load_topics(topics_csv: Path, row_start: int = 2):
    topics = []
    with topics_csv.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):  # First data line starts at 2.
            if i < row_start:
                continue
            topics.append({"main_topic": row["main_topic"], "sub_topic": row["sub_topic"]})
    return topics

def main():
    parser = argparse.ArgumentParser(description="Generate EEG stimuli for all topics")
    parser.add_argument("--topics_csv", type=str, required=True)
    parser.add_argument("--result_csv", type=str, default="data/generated/text.csv")
    parser.add_argument("--C", type=int, default=170)  # Character count target.
    parser.add_argument("--N", type=int, default=4)  # Samples per topic row.
    parser.add_argument("--row_start", type=int, default=2)
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: Set OPENAI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)
    topics = load_topics(Path(args.topics_csv), row_start=args.row_start)
    out_path = Path(args.result_csv)
    ensure_headers(out_path)

    for t in topics:
        mt, st = t["main_topic"], t["sub_topic"]
        print(f"### {mt} / {st} ###")
        for i in range(args.N):
            sample = call_model(mt, st, args.C)
            ans_idx = sample["answer_index"]
            answer_text = sample["options"][ans_idx - 1]
            record = {
                "main_topic": mt,
                "sub_topic": st,
                "C": args.C,
                "utterance": sample["utterance"],
                "question": sample["question"],
                "options": sample["options"],
                "answer": answer_text,
                "answer_index": ans_idx,
                "length_no_space": nospace_len(sample["utterance"]),
            }
            append_row(out_path, record)
            print(f"  [{i+1}/{args.N}] len={record['length_no_space']} saved")

if __name__ == "__main__":
    main()
