import json
import re

INPUT_FILE = "datasets/gsingh1/all.json"
KEYWORD_FILE = "datasets/keywords.json"
OUTPUT_FILE = "datasets/gsingh1/gsingh1-records.json"


def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 30]


def classify_topic(sentence, keywords):
    s = sentence.lower()

    scores = {}

    for topic, words in keywords.items():
        score = 0
        for w in words:
            if w.replace("_", " ") in s:
                score += 1
        scores[topic] = score

    best_topic = max(scores, key=scores.get)

    if scores[best_topic] == 0:
        return "unknown"

    return best_topic

def clean_text(text):

    text = text.replace("\n", " ")

    # remove markdown symbols
    text = re.sub(r'[#*]', '', text)

    # remove horizontal rules like ---
    text = re.sub(r'-{2,}', ' ', text)

    # collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# load dataset
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# load keywords
with open(KEYWORD_FILE, "r", encoding="utf-8") as f:
    keywords = json.load(f)

counter_mistral = 0
counter_gpt = 0
counter_human = 0
counter_gemma = 0
counter_llama = 0


records = []

for row in data:

    sources = {
        "human": row.get("Human_story"),
        "gemma": row.get("gemma-2-9b"),
        "mistral": row.get("mistral-7B"),
        "llama": row.get("llama-8B"),
        "chatgpt": row.get("GPT_4-o"),
    }

    for model, text in sources.items():

        if not text:
            continue

        text = clean_text(text)

        sentences = split_sentences(text)

        if model == "human":
            counter_human += 1
        elif model == "mistral":
            counter_mistral += 1
        elif model == "llama":
            counter_llama += 1
        elif model == "chatgpt":
            counter_gpt += 1
        elif model == "gemma":
            counter_gemma += 1

        for s in sentences:

            topic = classify_topic(s, keywords)

            if topic == "unknown":
                continue

            if not 80 < len(s) < 150:
                continue

            record = {
                "model": model,
                "text": s,
                "topic": topic,
                "origin": "gsingh1",
                "length": len(s)
            }

            records.append(record)


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)


print("Sentences created:", len(records))
print("gpt: ", counter_gpt)
print("mistral: ", counter_mistral)
print("gemini: ", counter_gemma)
print("llama: ", counter_llama)
print("human: ", counter_human)