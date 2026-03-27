import json
from collections import Counter

INPUT_FILE = "datasets/records_long.json"


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_counts = Counter()

    source_counts = Counter()

    for item in data:
        source = item.get("origin", "UNKNOWN")
        model = item.get("model", "UNKNOWN")
        source_counts[source] += 1
        model_counts[model] += 1


    print("Different sources and their counts:")
    for source, count in source_counts.items():
        print(f"{source}: {count}")
    print(f"\nTotal different sources: {len(source_counts)}")

    print(f"\n\nTotal sentences: {len(data)}")

    print("\n\nDifferent models and their counts:")
    for model, count in model_counts.items():
        print(f"{model}: {count}")


    print(f"\nTotal different models: {len(model_counts)}")


if __name__ == "__main__":
    main()