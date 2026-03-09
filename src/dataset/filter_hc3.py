import json

# Load keywords from keywords.json
keywords_file_path = "keywords.json"
with open(keywords_file_path) as f:
    keywords_data = json.load(f)

# Flatten all keywords from all categories into a single list
KEYWORDS = []
for category, keywords in keywords_data.items():
    KEYWORDS.extend(keywords)

hc3_file_path = "../../datasets/hc3/all.jsonl"
hc3_content = []
with open(hc3_file_path) as f:
    for line in f:
        record = json.loads(line)
        
        # Check if any keyword is present in question, human_answer, or chatgpt_answer
        if KEYWORDS:
            question = record.get('question', '').lower()
            human_answer = record.get('human_answer', '').lower()
            chatgpt_answer = record.get('chatgpt_answer', '').lower()
            
            has_keyword = any(
                keyword.lower() in question or 
                keyword.lower() in human_answer or 
                keyword.lower() in chatgpt_answer
                for keyword in KEYWORDS
            )
            
            if has_keyword:
                hc3_content.append(record)
        else:
            # If no keywords defined, keep all records
            hc3_content.append(record)

# Write filtered content to hc3-filtered.json
output_file_path = "hc3-filtered.json"
with open(output_file_path, 'w') as f:
    json.dump(hc3_content, f, indent=2)

print(f"Filtered {len(hc3_content)} records and saved to {output_file_path}")

    