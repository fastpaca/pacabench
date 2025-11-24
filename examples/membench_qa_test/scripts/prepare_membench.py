import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: str | None = None


class MemBenchInput(BaseModel):
    case_id: str
    input: str
    history: list[Message]
    choices: dict[str, str] | None = None


class MemBenchOutput(BaseModel):
    case_id: str
    question: str
    ground_truth: str | None
    choices: dict[str, str]
    history: list[Message]
    context_text: str
    answer_text: str | None = None
    ground_truth_text: str | None = None


def _normalize_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _normalize_choices(raw_choices: Any) -> dict[str, str]:
    if not raw_choices:
        return {}

    normalized: dict[str, str] = {}

    if isinstance(raw_choices, dict):
        items = raw_choices.items()
    elif isinstance(raw_choices, list):
        items = enumerate(raw_choices)
    else:
        return {}

    for key, value in items:
        if key is None:
            continue
        key_str = str(key).strip()
        if not key_str:
            continue

        # Normalize key to uppercase letter (A, B, C, D...)
        # If key is "0", "1", etc., map to A, B...
        normalized_key = chr(65 + int(key_str)) if key_str.isdigit() else key_str[0].upper()

        normalized[normalized_key] = _normalize_text(value)
    return normalized


def _extract_expected_choice(qa: dict[str, Any], choices: dict[str, str]) -> str | None:
    raw_expected = qa.get("ground_truth") or qa.get("answer")
    if isinstance(raw_expected, list):
        raw_expected = raw_expected[0] if raw_expected else None

    expected_text = _normalize_text(raw_expected)
    if not expected_text:
        return None

    # Case 1: Expected is "A", "B", etc.
    letter = expected_text[0].upper()
    if letter in choices:
        return letter

    # Case 2: Expected is the text of the answer
    lower_expected = expected_text.lower()
    for key, value in choices.items():
        if lower_expected == value.lower():
            return key

    return expected_text or None


def _normalize_history(message_list: list[Any]) -> list[Message]:
    normalized: list[Message] = []

    # Flatten if list of lists
    flat_list = []
    for item in message_list:
        if isinstance(item, list):
            flat_list.extend(item)
        else:
            flat_list.append(item)

    for msg in flat_list:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role")
        content = msg.get("content")
        timestamp = msg.get("timestamp") or msg.get("time")

        # Handle alternative format
        if not role or not content:
            u = msg.get("user_message") or msg.get("user")
            a = msg.get("assistant_message") or msg.get("assistant")
            if u:
                normalized.append(
                    Message(
                        role="user", content=str(u), timestamp=str(timestamp) if timestamp else None
                    )
                )
            if a:
                normalized.append(
                    Message(
                        role="assistant",
                        content=str(a),
                        timestamp=str(timestamp) if timestamp else None,
                    )
                )
        else:
            # Map arbitrary roles to standard ones if needed, currently assuming compliant
            normalized.append(
                Message(
                    role=role, content=str(content), timestamp=str(timestamp) if timestamp else None
                )
            )
    return normalized


def main():
    dataset_path = os.environ.get("PACABENCH_DATASET_PATH")
    if not dataset_path:
        print("PACABENCH_DATASET_PATH not set, assuming current directory or skipping.")
        return

    base_path = Path(dataset_path)
    source_dir = base_path / "MemData" / "FirstAgent"
    output_file = base_path / "membench.jsonl"

    if not source_dir.exists():
        print(f"Source directory {source_dir} does not exist.")
        return

    print(f"Converting files from {source_dir} to {output_file}")

    count = 0
    with open(output_file, "w") as outfile:
        for json_file in glob.glob(str(source_dir / "*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                records = data.get("events") or data.get("multi_agent") or []
                for record in records:
                    tid = record.get("tid", "unknown")
                    qa = record.get("QA", {})
                    question = qa.get("question", "")
                    choices = _normalize_choices(qa.get("choices"))
                    expected_choice = _extract_expected_choice(qa, choices)

                    message_list = record.get("message_list", [])
                    history = _normalize_history(message_list)

                    # Construct plain text context for display/debugging
                    context_lines = [f"{msg.role.capitalize()}: {msg.content}" for msg in history]
                    history_text = "\n".join(context_lines)

                    out_record = MemBenchOutput(
                        case_id=f"{Path(json_file).stem}-{tid}",
                        question=question,
                        ground_truth=expected_choice or _normalize_text(qa.get("ground_truth")),
                        choices=choices,
                        history=history,
                        context_text=history_text,
                        answer_text=_normalize_text(qa.get("answer")),
                        ground_truth_text=_normalize_text(qa.get("ground_truth")),
                    )

                    outfile.write(out_record.model_dump_json() + "\n")
                    count += 1

            except Exception as e:
                print(f"Error processing {json_file}: {e}", file=sys.stderr)

    print(f"Converted {count} records.")


if __name__ == "__main__":
    main()
