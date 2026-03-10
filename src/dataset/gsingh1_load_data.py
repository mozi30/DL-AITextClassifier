from datasets import load_dataset
import pandas as pd

dataset = load_dataset("gsingh1-py/train")

df = dataset["train"].to_pandas()
df.to_json("datasets/gsingh1/all.json", orient="records", indent=2)

print(df.head())