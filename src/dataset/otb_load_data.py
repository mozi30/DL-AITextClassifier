from datasets import load_dataset
import pandas as pd

dataset = load_dataset("MLNTeam-Unical/OpenTuringBench", "in_domain")

df = dataset["models"].to_pandas()

df.to_json("datasets/otb/all.json", orient="records", indent=2)

print(df.head())
#
# import json
#
# records = []
#
# for row in dataset["models"]:
#
#     text = row["text"]
#
#     if len(text) < 80 or len(text) > 140:
#         continue
#
#     model = row["model"].lower()
#
#     record = {
#         "model": model,
#         "text": text,
#         "topic": "unknown",
#         "origin": "OpenTuringBench",
#         "length": len(text)
#     }
#
#     records.append(record)
#
# with open("datasets/openturingbench-records.json", "w") as f:
#     json.dump(records, f, indent=2)