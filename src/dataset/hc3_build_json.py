import json
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt_tab", quiet=True)

# Load keywords from keywords.json
keywords_file_path = "../../datasets/keywords.json"
with open(keywords_file_path) as f:
    keywords_data = json.load(f)

# Build a map of category -> lowercase keywords
category_keywords = {
    category: [keyword.lower() for keyword in keywords]
    for category, keywords in keywords_data.items()
}

hc3_file_path = "../../datasets/hc3/all.jsonl"
hc3_content = []
with open(hc3_file_path) as f:
    for line in f:
        record = json.loads(line)
        question = record.get("question", "").lower()

        # HC3 records usually store answer arrays; support both arrays and strings.
        human_answers = record.get("human_answers", [])
        chatgpt_answers = record.get("chatgpt_answers", [])

        if isinstance(human_answers, list):
            human_text = " ".join(str(x) for x in human_answers).lower()
        else:
            human_text = str(human_answers).lower()

        if isinstance(chatgpt_answers, list):
            chatgpt_text = " ".join(str(x) for x in chatgpt_answers).lower()
        else:
            chatgpt_text = str(chatgpt_answers).lower()

        searchable_text = f"{question} {human_text} {chatgpt_text}"

        matched_categories = []
        for category, keywords in category_keywords.items():
            if any(keyword in searchable_text for keyword in keywords):
                matched_categories.append(category)

        if matched_categories:
            record["topic"] = matched_categories
            hc3_content.append(record)

# Write filtered content to hc3-filtered.json
# output_file_path = "hc3-filtered.json"
# with open(output_file_path, "w") as f:
#     json.dump(hc3_content, f, indent=2)

# print(f"Filtered {len(hc3_content)} records and saved to {output_file_path}")


def answer_to_json(answer, model, topic):
    sentences = sent_tokenize(str(answer))
    json_objects = []
    for sentence in sentences:
        if len(sentence) < 50 or len(sentence) > 200:
            continue
        json_object = {
            "model": model,
            "text": sentence,
            "topic": topic,
            "origin": "HC3",
            "length": len(sentence),
        }
        json_objects.append(json_object)
    return json_objects


output_record = []
i = 0
for line in hc3_content:
    print("Process Line " + str(i+1))
    i=i+1
    topics = line.get("topic", [])
    topic_value = topics[0] if topics else "unknown"

    human_answers = line.get("human_answers", [])
    chatgpt_answers = line.get("chatgpt_answers", [])

    if isinstance(human_answers, str):
        human_answers = [human_answers]
    if isinstance(chatgpt_answers, str):
        chatgpt_answers = [chatgpt_answers]

    for answer in human_answers:
        output_record.extend(answer_to_json(answer, "human", topic_value))

    for answer in chatgpt_answers:
        output_record.extend(answer_to_json(answer, "chatgpt", topic_value))
    break

output_record_path = "../../datasets/hc3/hc3-records.json"
with open(output_record_path, "w") as f:
    json.dump(output_record, f, indent=2)

print(f"Converted {len(output_record)} sentence records and saved to {output_record_path}")

    