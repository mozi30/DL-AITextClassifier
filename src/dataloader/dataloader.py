import json
import random
import unicodedata
import re
from pathlib import Path
from collections import defaultdict
import numpy as np

MODEL_MAP = {
    "human": 0,
    "chatgpt": 1,
    "gemma": 2,
    "mistral": 3,
    "llama": 4
}

class SentenceDataLoader:
    def __init__(self, samples: list, seed: int = 42):
        self.samples = samples
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]

    def get_random_item(self):
        if len(self.samples) == 0:
            raise ValueError("Dataset is empty")
        idx = self.rng.integers(len(self.samples))
        return self.samples[idx]
    
    def get_model_name(model_id: int):
        ID_TO_MODEL = {v: k for k, v in MODEL_MAP.items()}
        return  ID_TO_MODEL[model_id]
    
    def _split_into_words(sentence: str):
        return re.findall(r"\b\w+\b", sentence.lower())
    
    def get_most_common_words(self, top_n=20000):
        word_count = {}
        for sample in self.samples:
            sentence = sample["text"]
            words = SentenceDataLoader._split_into_words(sentence)
            for word in words:
                word_count[word] = word_count.get(word, 0) + 1
        # Sort by frequency, descending
        most_common = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        return most_common[:top_n]
            
            


def normalize_text(text: str):
        # unicode normalization
        text = unicodedata.normalize("NFKC", text)
        # normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # collapse whitespace
        text = re.sub(r"[ \t]+", " ", text)
        # collapse multiple newlines
        text = re.sub(r"\n+", "\n", text)
        return text.strip()


#Written by chatgpt
def balanced_sample_with_fallback(samples, total_size, seed=42):
    rng = random.Random(seed)

    # 1. Group by model
    groups = defaultdict(list)
    for sample in samples:
        groups[sample["model"]].append(sample)

    model_names = list(groups.keys())
    num_models = len(model_names)

    if num_models == 0:
        return []

    # Shuffle each group so sampling is random
    for model in model_names:
        rng.shuffle(groups[model])

    # If requested more than available, just return all shuffled
    if total_size >= len(samples):
        all_samples = samples[:]
        rng.shuffle(all_samples)
        return all_samples

    # 2. Ideal equal share
    base_quota = total_size // num_models

    selected = []
    taken_per_model = {}

    # 3. First pass: take up to base_quota from each model
    for model in model_names:
        take = min(base_quota, len(groups[model]))
        selected.extend(groups[model][:take])
        taken_per_model[model] = take

    # 4. Count remaining slots
    remaining = total_size - len(selected)

    # 5. Build leftover pool from models that still have more
    if remaining > 0:
        leftovers = []
        for model in model_names:
            already_taken = taken_per_model[model]
            leftovers.extend(groups[model][already_taken:])

        rng.shuffle(leftovers)
        selected.extend(leftovers[:remaining])

    # Final shuffle so output is mixed
    rng.shuffle(selected)
    return selected


class SentenceDataModule:

    def __init__(self, record_path: str = "../../datasets/records.json", models: list | None = None, normalize: bool = True, size: int | None = None , split: tuple[int, int, int] = (70, 20, 10), seed = 42):
        self.record_path = Path(record_path)
        self.normalize = normalize
        self.size = size
        self.split = split
        self.seed = seed
        self.models = models
        
        self.train_samples = []
        self.val_samples = []
        self.test_samples = []
        if split[0] + split[1] + split[2] != 100:
            raise ValueError(f"Split ratios must sum to 100, got {split}.")
        self._load()

    # ----------------------------
    # Loading
    # ----------------------------
    
    def _load(self):
        """Load dataset from file."""
            
        with open(self.record_path, "r", encoding="utf-8") as f:
            json_object = json.load(f)

        records_samples = []
        sample_id = 0
        for raw in json_object:
            # Map model name -> numeric id
            model = raw["model"]
            if self.models != None:
                if model not in self.models:
                    continue
            model_id = MODEL_MAP[model]

            text = raw["text"]

            if self.normalize:
                text = normalize_text(text)

            topic = raw["topic"]
            origin = raw["origin"]

            length = len(text)

            sample = {
                "id": sample_id,
                "model": model_id,
                "text": text,
                "topic": topic,
                "origin": origin,
                "length": length,
            }
            records_samples.append(sample)
            sample_id = sample_id + 1
        if self.size is None:
            self.size = len(records_samples)
        
        samples = balanced_sample_with_fallback(records_samples, self.size, self.seed)
        num_samples = len(samples)
        print("Sample in dataset " + str(num_samples))

        train_end = int(num_samples * float(self.split[0])/100)
        val_end = int(num_samples * (self.split[0] + self.split[1]) / 100)
        # val_end = int(num_samples * float(self.split[0])/100 + float(self.split[1])/100)

        self.train_samples = samples[:train_end]
        self.val_samples = samples[train_end:val_end]
        self.test_samples = samples[val_end:]
        
    def get_train_loader(self):
        return SentenceDataLoader(self.train_samples) 
        
    def get_val_loader(self):
        return SentenceDataLoader(self.val_samples)
    
    def get_test_loader(self):
        return SentenceDataLoader(self.test_samples)
    
    def get_model_name(model_id: int):
        ID_TO_MODEL = {v: k for k, v in MODEL_MAP.items()}
        return  ID_TO_MODEL[model_id]
    
    def get_most_common_words(self, top_n=20000):
        word_count = {}
        for sample in self.train_samples:
            sentence = sample["text"]
            words = SentenceDataLoader._split_into_words(sentence)
            for word in words:
                word_count[word] = word_count.get(word, 0) + 1
        # Sort by frequency, descending
        most_common = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        return most_common[:top_n]

# if __name__ == "__main__":
#     dataloader = SentenceDataModule(size=200000)
    
#     test_loader = dataloader.get_test_loader()
    
#     for i in range(10):
#         sample = test_loader[i]
#         print("Model: " + str(sample["model"]) + " Text: " +  sample["text"])