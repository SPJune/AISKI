import pandas as pd
import json

# File paths
INPUT_CSV = 'data/generated/topics.csv'
GROUPED_JSON = 'data/generated/grouped_topics.json'
OUTPUT_CSV = 'data/generated/topics_representative.csv'

# 1) Load data
df = pd.read_csv(INPUT_CSV)

# 2) Load representative topic groups
with open(GROUPED_JSON, 'r', encoding='utf-8') as f:
    grouped_topics = json.load(f)

# 3) Reverse mapping: original topic -> representative topic
topic_to_representative = {}
for representative, topic_list in grouped_topics.items():
    for topic in topic_list:
        topic_to_representative[topic] = representative

# 4) Map main_topic to representative topic
df['main_topic'] = df['main_topic'].apply(
    lambda topic: topic_to_representative.get(topic, topic)  # Keep original value if no mapping.
)

# 5) Save
df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
print(f"Saved representative topic CSV to '{OUTPUT_CSV}'.")
