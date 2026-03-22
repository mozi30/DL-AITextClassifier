import argparse
import json
import re
from pathlib import Path


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s and s.strip()]


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def build_passages(sentences: list[str], min_words: int, max_words: int) -> list[str]:
    passages: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        s_words = word_count(sentence)
        if s_words == 0:
            continue

        # Skip single sentences that are themselves longer than max_words.
        if s_words >= max_words:
            continue

        # If adding this sentence would exceed the max, close out the current
        # passage (if it's long enough) and start a new one from this
        # sentence.
        if current_words + s_words > max_words:
            if current_words >= min_words:
                passages.append(" ".join(current))
            current = [sentence]
            current_words = s_words
            continue

        # Otherwise, keep growing the current passage.
        current.append(sentence)
        current_words += s_words

    # Flush any remaining passage that meets the minimum size.
    if current_words >= min_words:
        passages.append(" ".join(current))

    return passages


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


def build_records_from_multiround(
    data: list[dict],
    keywords: dict[str, list[str]],
    min_words: int,
    max_words: int,
) -> list[dict]:
    """Build records from Claude multiround chat dataset.

    - Filter conversations by presence of domain keywords anywhere in the
      conversation (using classify_topic on the full conversation text).
    - Take only messages written by Claude ("from": "gpt").
    - Normalize/clean text like other builders and split into passages of
      length between min_words and max_words.
    - Label the model consistently with other code ("chatgpt").
    """

    records: list[dict] = []

    for row in data:
        conversations = row.get("conversations")
        if not isinstance(conversations, list):
            continue

        # Build full conversation text for topic classification
        full_parts: list[str] = []
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            value = turn.get("value")
            if isinstance(value, str):
                full_parts.append(clean_text(value))

        if not full_parts:
            continue

        full_text = " ".join(full_parts)
        topic = classify_topic(full_text, keywords)
        if topic == "unknown":
            # Filter out conversations without any domain keywords
            continue

        # Claude responses are tagged as "gpt" in this dataset; map them into
        # the existing "chatgpt" bucket used elsewhere.
        model = normalize_model("anthropic") or "chatgpt"

        for turn in conversations:
            if not isinstance(turn, dict):
                continue

            if turn.get("from") != "gpt":
                # Only use Claude's responses
                continue

            value = turn.get("value")
            if not isinstance(value, str):
                continue

            answer = clean_text(value)
            if not answer:
                continue

            passages = build_passages(split_sentences(answer), min_words, max_words)
            for passage in passages:
                if not passage:
                    continue
                records.append(
                    {
                        "model": "anthropic",
                        "text": passage,
                        "topic": topic,
                        "origin": "claude_multiround_chat_30k",
                        # Store snippet length in words so it directly
                        # reflects the min_words/max_words constraints.
                        "length": word_count(passage),
                    }
                )

    return records


def normalize_model(model_name: str | None) -> str | None:
    if not model_name:
        return None
    m = model_name.lower()

    # Anthropic/Claude answers are AI-generated; map into the existing chatgpt label space.
    if "anthropic" in m or "claude" in m:
        return "chatgpt"
    if "gpt" in m or "openai" in m:
        return "chatgpt"
    if "gemma" in m or "gemini" in m or "google" in m:
        return "gemma"
    if "mistral" in m:
        return "mistral"
    if "llama" in m:
        return "llama"
    if "human" in m:
        return "human"
    return None


def pick_topic(row_categories, question: str, answer: str, keywords: dict[str, list[str]]) -> str:
    if isinstance(row_categories, list):
        for item in row_categories:
            if isinstance(item, str):
                t = item.strip().lower()
                if t in keywords:
                    return t

    searchable = f"{question} {answer}"
    return classify_topic(searchable, keywords)


def build_anthropic_records(
    input_path: Path,
    keywords_path: Path,
    output_path: Path,
    min_words: int,
    max_words: int,
) -> list[dict]:
    """Programmatic entry point for building anthropic-records.json.

    This is shared between the CLI (main) and other dataset builders
    (e.g. build_all_records) so they can trigger Anthropic snippet
    generation without shelling out to a separate process.
    """

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    with keywords_path.open("r", encoding="utf-8") as f:
        keywords = json.load(f)

    records: list[dict] = []

    # Auto-detect whether we're dealing with the original single-turn
    # Anthropic format (question/answer) or the Claude multiround
    # conversation format ("conversations" key).
    if isinstance(data, list) and data and isinstance(data[0], dict) and "conversations" in data[0]:
        records = build_records_from_multiround(data, keywords, min_words, max_words)
    else:
        for row in data:
            model = normalize_model(row.get("model"))
            if model is None:
                continue

            question = clean_text(str(row.get("question", "")))
            answer = clean_text(str(row.get("answer", "")))
            if not answer:
                continue

            topic = pick_topic(row.get("categories"), question, answer, keywords)
            if topic == "unknown":
                continue

            passages = build_passages(split_sentences(answer), min_words, max_words)
            for passage in passages:
                records.append(
                    {
                        "model": model,
                        "text": passage,
                        "topic": topic,
                        "origin": "anthropic",
                        # Use word-level length for consistency with other
                        # builders and with the configured word bounds.
                        "length": word_count(passage),
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Anthropic records: {len(records)}")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Build anthropic-records.json from datasets/anthropic/all.json")
    # Default to the Claude multiround chat dataset, but keep support for
    # the original single-turn all.json format when explicitly requested.
    parser.add_argument("--input", default="../../datasets/anthropic/claude_multiround_chat_30k.json")
    parser.add_argument("--keywords", default="../../datasets/keywords.json")
    parser.add_argument("--output", default="../../datasets/anthropic/anthropic-records.json")
    parser.add_argument("--records-long", default="../../datasets/records_long.json")
    parser.add_argument(
        "--skip-records-long-update",
        action="store_true",
        help="Only build anthropic-records.json and do not merge into records_long.json",
    )
    parser.add_argument("--min-words", type=int, default=80)
    parser.add_argument("--max-words", type=int, default=200)
    args = parser.parse_args()

    input_path = Path(args.input)
    keywords_path = Path(args.keywords)
    output_path = Path(args.output)
    records_long_path = Path(args.records_long)

    records = build_anthropic_records(
        input_path=input_path,
        keywords_path=keywords_path,
        output_path=output_path,
        min_words=args.min_words,
        max_words=args.max_words,
    )

    if not args.skip_records_long_update:
        existing_records: list[dict] = []
        if records_long_path.exists():
            with records_long_path.open("r", encoding="utf-8") as f:
                existing_records = json.load(f)

        seen = {
            (
                str(r.get("model", "")),
                str(r.get("text", "")),
                str(r.get("topic", "")),
                str(r.get("origin", "")),
            )
            for r in existing_records
            if isinstance(r, dict)
        }

        added = 0
        for row in records:
            key = (
                str(row.get("model", "")),
                str(row.get("text", "")),
                str(row.get("topic", "")),
                str(row.get("origin", "")),
            )
            if key in seen:
                continue
            existing_records.append(row)
            seen.add(key)
            added += 1

        records_long_path.parent.mkdir(parents=True, exist_ok=True)
        with records_long_path.open("w", encoding="utf-8") as f:
            json.dump(existing_records, f, indent=2, ensure_ascii=False)

        print(f"records_long updated: +{added} (total={len(existing_records)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
