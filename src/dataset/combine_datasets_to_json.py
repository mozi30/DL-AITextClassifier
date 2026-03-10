import json
import re

INPUT_FILE_HC3 = "datasets/hc3/hc3-records.json"
INPUT_FILE_GSINGH1 = "datasets/gsingh1/gsingh1-records.json"
INPUT_FILE_OTB = "datasets/otb/otb-records.json"
OUTPUT_FILE = "datasets/records.json"


with open(INPUT_FILE_HC3, "r", encoding="utf-8") as f:
    data_hc3 = json.load(f)

with open(INPUT_FILE_GSINGH1, "r", encoding="utf-8") as f:
    data_gsingh1 = json.load(f)

with open(INPUT_FILE_OTB, "r", encoding="utf-8") as f:
    data_otb = json.load(f)


combined = data_hc3 + data_gsingh1 + data_otb

counter_mistral = 0
counter_gpt = 0
counter_human = 0
counter_gemma = 0
counter_llama = 0

for row in combined:
    model = row["model"]
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


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(combined, f, indent=2, ensure_ascii=False)

print("gpt: ", counter_gpt)
print("mistral: ", counter_mistral)
print("gemma: ", counter_gemma)
print("llama: ", counter_llama)
print("human: ", counter_human)
print("HC3 samples:", len(data_hc3))
print("GSINGH1 samples:", len(data_gsingh1))
print("OTB samples:", len(data_otb))
print("Total samples:", len(combined))