import os
import csv
import json
import time
from typing import List, Tuple
from openai import OpenAI

# Load API key from environment.
API_KEY = os.getenv("OPENAI_API_KEY")
print("API_KEY available:", API_KEY is not None)
client = OpenAI(api_key=API_KEY)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

SYSTEM_PROMPT = """
당신은 실험용 음성 자극을 위한 주제를 제안하는 어시스턴트입니다.
요구사항:
- 주제는 반드시 "상식, 지식 전달"이 아닌 "개인적인 경험", "일기 형식", 또는 "가상의 이야기"에 적합해야 합니다.
- main_topic과 그에 속하는 구체적인 sub_topic을 생성하세요.
- main_topic은 중범위 주제, sub_topic은 구체적인 일상/경험/감정/상황 중심으로 작성하세요.
- sub_topic은 주제별로 다양하게 5개씩 제안하세요.
- 각 주제는 중복되지 않게 하세요.
- 주제는 모두 한국어로 작성하세요.
출력 형식(JSON):
[
  {"main_topic": "topic1", "sub_topics": ["sub1", "sub2", "sub3", "sub4", "sub5"]},
  ...
]
"""

def generate_topics(n_blocks: int = 10) -> List[Tuple[str, str]]:
    all_pairs = set()

    for i in range(n_blocks):
        print(f"Generating request {i+1}/{n_blocks}...")
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0.9,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"{i+1}번째 주제 묶음을 생성해줘."}
                ],
                max_tokens=1000
            )

            content = response.choices[0].message.content.strip()

            # Remove markdown code fences.
            if content.startswith("```"):
                content = content.strip("`").strip()
                content = content.replace("json", "").strip()

            parsed = json.loads(content)

            for item in parsed:
                main = item['main_topic'].strip()
                for sub in item['sub_topics']:
                    sub = sub.strip()
                    all_pairs.add((main, sub))

            time.sleep(1)  # Avoid rate-limit spikes.

        except Exception as e:
            print(f"[ERROR] Failed to generate block {i+1}: {e}")
            continue

    return list(all_pairs)

def save_to_csv(pairs: List[Tuple[str, str]], filename="data/generated/topics.csv"):
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["main_topic", "sub_topic"])
        for main, sub in sorted(pairs):
            writer.writerow([main, sub])
    print(f"Saved {len(pairs)} pairs -> {filename}")

if __name__ == "__main__":
    topic_pairs = generate_topics(n_blocks=20)
    save_to_csv(topic_pairs)
