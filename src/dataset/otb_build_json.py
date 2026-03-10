import json
import re

INPUT_FILE = "datasets/otb/all.json"
KEYWORD_FILE = "datasets/keywords.json"
OUTPUT_FILE = "datasets/otb/otb-records.json"


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


def map_model(model_name):
    if not model_name:
        return None

    m = model_name.lower()

    if "gemma" in m or "gemini" in m or "google" in m:
        return "gemma"

    if "mistral" in m:
        return "mistral"

    if "llama" in m:
        return "llama"

    if "gpt" in m or "openai" in m:
        return "chatgpt"

    if "human" in m:
        return "human"

    return None


for row in data:

    model_raw = row["model"]
    model = map_model(model_raw)

    if model is None:
        continue

    text = row.get("content")

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
            "origin": "otb",
            "length": len(s)
        }

        records.append(record)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

print("Sentences created:", len(records))
print("gpt: ", counter_gpt)
print("mistral: ", counter_mistral)
print("gemma: ", counter_gemma)
print("llama: ", counter_llama)
print("human: ", counter_human)
