import argparse
import json
import re
from pathlib import Path

# Support both package execution (python -m src.dataset.build_all_records)
# and direct script execution from this folder (python build_all_records.py).
try:  # package context
    from .anthropic_build_json import build_anthropic_records
except ImportError:  # script context, no known parent package
    from anthropic_build_json import build_anthropic_records


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s and s.strip()]


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def split_text_by_words(text: str, max_words: int) -> list[str]:
    """Split text into chunks of at most max_words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    return chunks


def build_passages(sentences: list[str], min_words: int, max_words: int) -> list[str]:
    passages: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        s_words = word_count(sentence)
        if s_words == 0:
            continue

        # If a single sentence is too long, split it into word chunks
        if s_words >= max_words:
            # First, save current passage if it meets criteria
            if min_words <= current_words <= max_words:
                passages.append(" ".join(current))
            current = []
            current_words = 0
            
            # Split the long sentence into chunks
            chunks = split_text_by_words(sentence, max_words)
            for chunk in chunks:
                chunk_words = word_count(chunk)
                if min_words <= chunk_words <= max_words:
                    passages.append(chunk)
            continue

        # If adding this sentence would exceed max_words, save current and start new
        if current_words + s_words > max_words:
            if min_words <= current_words <= max_words:
                passages.append(" ".join(current))
            current = [sentence]
            current_words = s_words
            continue

        # Add sentence to current passage
        current.append(sentence)
        current_words += s_words

    # Don't forget the last passage
    if min_words <= current_words <= max_words:
        passages.append(" ".join(current))

    return passages


def refine_passages(passages: list[str], min_words: int, max_words: int) -> list[str]:
    """Further split passages that exceed max_words into multiple chunks with varied sizes for better distribution."""
    refined: list[str] = []
    for passage in passages:
        p_words = word_count(passage)
        if p_words > max_words:
            words = passage.split()
            # Calculate number of chunks needed
            num_chunks = (p_words + max_words - 1) // max_words
            
            # Distribute words across chunks with varying target sizes
            # This creates better distribution across the min-max range
            target_sizes = []
            for i in range(num_chunks):
                # Vary target size across the range: min_words to max_words
                ratio = i / max(1, num_chunks - 1) if num_chunks > 1 else 0.5
                target_size = int(min_words + (max_words - min_words) * ratio)
                target_sizes.append(target_size)
            
            # Create chunks based on target sizes
            word_idx = 0
            for target_size in target_sizes:
                if word_idx >= len(words):
                    break
                # Take roughly target_size words, but allow some flexibility
                chunk_end = min(word_idx + target_size, len(words))
                chunk = " ".join(words[word_idx:chunk_end])
                chunk_words = word_count(chunk)
                if min_words <= chunk_words <= max_words:
                    refined.append(chunk)
                word_idx = chunk_end
            
            # Handle any remaining words that didn't fit into target chunks
            if word_idx < len(words):
                remainder = " ".join(words[word_idx:])
                remainder_words = word_count(remainder)
                if min_words <= remainder_words <= max_words:
                    refined.append(remainder)
        else:
            refined.append(passage)
    return refined


def clean_text(text: str) -> str:
    text = (text or "").replace("\n", " ")
    text = re.sub(r"[#*]", "", text)
    text = re.sub(r"-{2,}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def classify_topic(text: str, keywords: dict[str, list[str]]) -> str:
    s = text.lower()
    scores: dict[str, int] = {}

    for topic, words in keywords.items():
        score = 0
        for w in words:
            if w.replace("_", " ") in s:
                score += 1
        scores[topic] = score

    best_topic = max(scores, key=scores.get) if scores else "unknown"
    if not scores or scores[best_topic] == 0:
        return "unknown"
    return best_topic


def map_model(model_name: str | None) -> str | None:
    if not model_name:
        return None

    m = model_name.lower()

    if "gemma" in m or "gemini" in m or "google" in m:
        return "gemma"
    # if "mistral" in m:
    #     return "mistral"
    if "llama" in m:
        return "llama"
    if "gpt" in m or "openai" in m:
        return "chatgpt"
    if "anthropic" in m or "claude" in m:
        return "anthropic"
    if "human" in m:
        return "human"

    return None


def load_keywords(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def download_otb_all_json(output_path: Path) -> None:
    from datasets import load_dataset
    import pandas as pd

    dataset = load_dataset("MLNTeam-Unical/OpenTuringBench", "in_domain")
    df = pd.concat([dataset["train"].to_pandas(), dataset["val"].to_pandas()], ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient="records", indent=2)


def download_gsingh1_all_json(output_path: Path) -> None:
    from datasets import load_dataset

    dataset = load_dataset("gsingh1-py/models")
    df = dataset["models"].to_pandas()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient="records", indent=2)


def build_hc3_records(
    input_path: Path,
    keywords: dict[str, list[str]],
    output_path: Path,
    min_words: int,
    max_words: int,
) -> list[dict]:
    output_records: list[dict] = []

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            question = str(row.get("question", ""))

            human_answers = row.get("human_answers", [])
            chatgpt_answers = row.get("chatgpt_answers", [])

            if isinstance(human_answers, str):
                human_answers = [human_answers]
            if isinstance(chatgpt_answers, str):
                chatgpt_answers = [chatgpt_answers]

            human_text = " ".join(str(x) for x in human_answers)
            chatgpt_text = " ".join(str(x) for x in chatgpt_answers)
            searchable = f"{question} {human_text} {chatgpt_text}"

            topic = classify_topic(searchable, keywords)
            if topic == "unknown":
                continue

            for answer in human_answers:
                text = clean_text(str(answer))
                passages = refine_passages(build_passages(split_sentences(text), min_words, max_words), min_words, max_words)
                for p in passages:
                    output_records.append(
                        {
                            "model": "human",
                            "text": p,
                            "topic": topic,
                            "origin": "HC3",
                            "length": word_count(p),
                        }
                    )

            for answer in chatgpt_answers:
                text = clean_text(str(answer))
                passages = refine_passages(build_passages(split_sentences(text), min_words, max_words), min_words, max_words)
                for p in passages:
                    output_records.append(
                        {
                            "model": "chatgpt",
                            "text": p,
                            "topic": topic,
                            "origin": "HC3",
                            "length": word_count(p),
                        }
                    )

    write_json(output_path, output_records)
    return output_records


def build_otb_records(
    input_path: Path,
    text_key: str,
    model_key: str,
    origin: str,
    keywords: dict[str, list[str]],
    output_path: Path,
    min_words: int,
    max_words: int,
    topic_key: str | None = None,
    skip_unknown_topic: bool = False,
    enforce_word_bounds: bool = True,
) -> list[dict]:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        
    records: list[dict] = []
    
    for row in data:
        model = map_model(row.get(model_key))
        if model is None:
            continue
        
        text = clean_text(str(row.get(text_key, "")))
        if enforce_word_bounds:
            passages = refine_passages(build_passages(split_sentences(text), min_words, max_words), min_words, max_words)
        else:
            if word_count(text) > max_words:
                passages = split_text_by_words(text, max_words)
            else:
                passages = [text] if text else []

        for p in passages:
            topic = "unknown"
            if topic_key:
                raw_topic = row.get(topic_key)
                if isinstance(raw_topic, list):
                    topic = str(raw_topic[0]).strip() if raw_topic else "unknown"
                elif isinstance(raw_topic, str):
                    topic = raw_topic.strip() if raw_topic.strip() else "unknown"
            if topic == "unknown":
                topic = classify_topic(p, keywords)
            if (not skip_unknown_topic) and topic == "unknown":
                continue
            records.append(
                {
                    "model": model,
                    "text": p,
                    "topic": topic,
                    "origin": origin,
                    "length": word_count(p),
                }
            )
    write_json(output_path, records)
    return records


def build_gsingh1_records(
    input_path: Path,
    keywords: dict[str, list[str]],
    output_path: Path,
    min_words: int,
    max_words: int,
) -> list[dict]:

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    records = []

    for row in data:

        prompt = str(row.get("prompt", ""))

        # --- HUMAN ---
        human_text = clean_text(row.get("Human_story", ""))
        passages = build_passages(split_sentences(human_text), min_words, max_words)

        for p in passages:
            topic = classify_topic(p, keywords)
            if topic == "unknown":
                continue

            records.append({
                "model": "human",
                "text": p,
                "topic": topic,
                "origin": "gsingh1",
                "length": len(p),
            })

        # --- MODELS ---
        for key, value in row.items():

            if key in ["prompt", "Human_story"]:
                continue

            model = map_model(key)
            if model is None:
                continue

            text = clean_text(str(value))
            passages = build_passages(split_sentences(text), min_words, max_words)

            for p in passages:
                topic = classify_topic(p, keywords)
                if topic == "unknown":
                    continue

                records.append({
                    "model": model,
                    "text": p,
                    "topic": topic,
                    "origin": "gsingh1",
                    "length": len(p),
                })

    write_json(output_path, records)
    return records


def summarize(records: list[dict], name: str) -> None:
    counter = {"human": 0, "chatgpt": 0, "mistral": 0, "gemma": 0, "llama": 0, "anthropic": 0}
    for r in records:
        model = r.get("model")
        if model in counter:
            counter[model] += 1

    print(f"{name} samples: {len(records)}")
    print(f"{name} human: {counter['human']}")
    print(f"{name} chatgpt: {counter['chatgpt']}")
    print(f"{name} mistral: {counter['mistral']}")
    print(f"{name} gemma: {counter['gemma']}")
    print(f"{name} llama: {counter['llama']}")
    print(f"{name} anthropic: {counter['anthropic']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build records JSON files from HC3/GSINGH1/OTB/Anthropic in one run.")
    parser.add_argument("--download-sources", action="store_true", help="Download source all.json files for OTB and GSINGH1.")
    parser.add_argument("--min-words", type=int, default=80)
    parser.add_argument("--max-words", type=int, default=200)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    datasets_dir = root / "datasets"

    keywords = load_keywords(datasets_dir / "keywords.json")

    hc3_jsonl = datasets_dir / "hc3" / "all.jsonl"
    gsingh1_all = datasets_dir / "gsingh1" / "all.json"
    otb_all = datasets_dir / "otb" / "all.json"
    anthropic_all = datasets_dir / "anthropic" / "all.json"

    hc3_records_path = datasets_dir / "hc3" / "hc3-records.json"
    gsingh1_records_path = datasets_dir / "gsingh1" / "gsingh1-records.json"
    otb_records_path = datasets_dir / "otb" / "otb-records.json"
    anthropic_records_path = datasets_dir /"anthropic" / "anthropic-records.json"
    combined_long_path = datasets_dir / "records_long.json"

    if args.download_sources:
        if otb_all.exists():
            print(f"Skipping OTB download, file already exists: {otb_all}")
        else:
            try:
                print("Downloading OTB source...")
                download_otb_all_json(otb_all)
                print(f"Saved {otb_all}")
            except Exception as exc:
                print(f"Warning: could not download OTB source: {exc}")

        if gsingh1_all.exists():
            print(f"Skipping GSINGH1 download, file already exists: {gsingh1_all}")
        else:
            try:
                print("Downloading GSINGH1 source...")
                download_gsingh1_all_json(gsingh1_all)
                print(f"Saved {gsingh1_all}")
            except Exception as exc:
                print(f"Warning: could not download GSINGH1 source: {exc}")

    if not hc3_jsonl.exists():
        raise FileNotFoundError(f"Missing HC3 source file: {hc3_jsonl}")

    if not otb_all.exists():
        raise FileNotFoundError(
            f"Missing OTB source file: {otb_all}. Use --download-sources or run src/dataset/otb_load_data.py first."
        )

    # For Anthropic, if the precomputed anthropic-records.json does not
    # exist yet, automatically build it using the same logic as the
    # anthropic_build_json script (typically from the Claude multiround
    # chat dataset).
    if not anthropic_records_path.exists():
        print("Anthropic records file missing, building it now ...")
        claude_multiround_path = datasets_dir / "anthropic" / "claude_multiround_chat_30k.json"
        build_anthropic_records(
            input_path=claude_multiround_path,
            keywords_path=datasets_dir / "keywords.json",
            output_path=anthropic_records_path,
            min_words=args.min_words,
            max_words=args.max_words,
        )

    print("Building HC3 records...")
    hc3_records = build_hc3_records(hc3_jsonl, keywords, hc3_records_path, args.min_words, args.max_words)

    gsingh1_records: list[dict] = []
    if gsingh1_all.exists():
        print("Building GSINGH1 records...")
        gsingh1_records = build_gsingh1_records(
            gsingh1_all,
            keywords,
            gsingh1_records_path,
            args.min_words,
            args.max_words,
        )
    else:
        print(f"Warning: skipping GSINGH1 build because source file is missing: {gsingh1_all}")

    print("Building OTB records...")
    otb_records = build_otb_records(
        otb_all,
        text_key="content",
        model_key="model",
        origin="OpenTuringBench",
        keywords=keywords,
        output_path=otb_records_path,
        min_words=args.min_words,
        max_words=args.max_words,
    )

    print("Loading Anthropic records ...")
    with anthropic_records_path.open("r", encoding="utf-8") as f:
        anthropic_records = json.load(f)
    
    
    combined = hc3_records + gsingh1_records + otb_records + anthropic_records 

    combined_long = [r for r in combined if args.min_words <= word_count(r["text"]) <= args.max_words]
    write_json(combined_long_path, combined_long)
    summarize(combined_long, "LONG")
    print(f"Wrote: {combined_long_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
