import glob
import json
import os
from pathlib import Path
from typing import Any


def _normalize_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _normalize_choices(raw_choices: Any) -> dict[str, str]:
    if not raw_choices:
        return {}

    if isinstance(raw_choices, dict):
        items = raw_choices.items()
    elif isinstance(raw_choices, list):
        items = enumerate(raw_choices)
    else:
        return {}

    normalized: dict[str, str] = {}
    for key, value in items:
        if key is None:
            continue
        key_str = str(key).strip()
        if not key_str:
            continue
        normalized_key = key_str[0].upper()
        normalized[normalized_key] = _normalize_text(value)
    return normalized


def _extract_expected_choice(qa: dict[str, Any], choices: dict[str, str]) -> str | None:
    raw_expected = qa.get("ground_truth") or qa.get("answer")
    if isinstance(raw_expected, list):
        raw_expected = raw_expected[0] if raw_expected else None

    expected_text = _normalize_text(raw_expected)
    if expected_text:
        letter = expected_text[0].upper()
        if letter in choices:
            return letter

        lower_expected = expected_text.lower()
        for key, value in choices.items():
            if lower_expected == value.lower():
                return key

    return expected_text or None


def main():
    dataset_path = os.environ.get("AGENTBENCH_DATASET_PATH")
    if not dataset_path:
        print("AGENTBENCH_DATASET_PATH not set, assuming current directory or skipping.")
        return

    base_path = Path(dataset_path)
    source_dir = base_path / "MemData" / "FirstAgent"
    output_file = base_path / "membench.jsonl"

    print(f"Converting files from {source_dir} to {output_file}")

    if not source_dir.exists():
        print(f"Source directory {source_dir} does not exist.")
        return

    with open(output_file, "w") as outfile:
        for json_file in glob.glob(str(source_dir / "*.json")):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)

                records = data.get("events") or data.get("multi_agent") or []
                for record in records:
                    tid = record.get("tid", "unknown")
                    qa = record.get("QA", {})
                    question = qa.get("question", "")
                    choices = _normalize_choices(qa.get("choices"))
                    expected_choice = _extract_expected_choice(qa, choices)

                    message_list = record.get("message_list", [])
                    context = []
                    for msg in message_list:
                        if isinstance(msg, dict):
                            u = msg.get("user_message") or msg.get("user")
                            a = msg.get("assistant_message") or msg.get("assistant")
                            if u:
                                context.append(f"User: {u}")
                            if a:
                                context.append(f"Assistant: {a}")

                    history_text = "\n".join(context)

                    out_record = {
                        "case_id": f"{Path(json_file).stem}-{tid}",
                        "question": question,
                        "ground_truth": expected_choice or _normalize_text(qa.get("ground_truth")),
                        "choices": choices,
                        "history": message_list,
                        "context_text": history_text,
                        "answer_text": _normalize_text(qa.get("answer")),
                        "ground_truth_text": _normalize_text(qa.get("ground_truth")),
                    }

                    outfile.write(json.dumps(out_record) + "\n")

            except Exception as e:
                print(f"Error processing {json_file}: {e}")


if __name__ == "__main__":
    main()
