from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

from .types import ConversationSample

MESSAGES_KEY = "message_list"
QA_KEY = "QA"


def normalize_messages(message_list: Sequence) -> List[dict]:
    """Flatten nested user/assistant messages into simple turn dictionaries."""

    def format_turn(role: str, text: str, meta: dict | None) -> dict:
        if not text:
            return {}
        suffix_parts = []
        if meta:
            time_val = meta.get("time")
            place_val = meta.get("place")
            if time_val:
                suffix_parts.append(f"time: {time_val}")
            if place_val:
                suffix_parts.append(f"place: {place_val}")
        suffix = f" ({'; '.join(suffix_parts)})" if suffix_parts else ""
        return {"role": role, "content": f"{text}{suffix}"}

    turns: List[dict] = []
    for block in message_list:
        sessions = block if isinstance(block, list) else [block]
        for message in sessions:
            if not isinstance(message, dict):
                turns.append({"role": "user", "content": str(message)})
                continue
            meta = {"time": message.get("time"), "place": message.get("place")}
            user_text = message.get("user_message") or message.get("user")
            assistant_text = message.get("assistant_message") or message.get("assistant")
            if user_text:
                turns.append(format_turn("user", user_text, meta))
            if assistant_text:
                turns.append(format_turn("assistant", assistant_text, meta))

    return [turn for turn in turns if turn.get("content")]


def iter_conversations(source_dir: Path, include: set[str] | None = None) -> Iterator[ConversationSample]:
    """Yield ConversationSample objects from raw JSON files."""

    for json_path in sorted(source_dir.glob("*.json")):
        if include and json_path.stem not in include:
            continue
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        yield from _extract_from_node(payload, json_path.stem, [])


def _extract_from_node(node, dataset_name: str, category_path: List[str]) -> Iterator[ConversationSample]:
    if isinstance(node, list):
        for item in node:
            if isinstance(item, dict) and MESSAGES_KEY in item and QA_KEY in item:
                yield _build_sample(item, dataset_name, category_path)
            else:
                yield from _extract_from_node(item, dataset_name, category_path)
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from _extract_from_node(value, dataset_name, category_path + [key])


def _build_sample(record: dict, dataset_name: str, category_path: List[str]) -> ConversationSample:
    qa = record.get(QA_KEY, {})
    conversation = normalize_messages(record.get(MESSAGES_KEY, []))
    sample_id = f"{dataset_name}::{record.get('tid', 'unknown')}"
    return ConversationSample(
        id=sample_id,
        source_file=dataset_name,
        category_path=category_path,
        tid=record.get("tid"),
        conversation=conversation,
        question=qa.get("question", ""),
        choices=qa.get("choices", {}),
        ground_truth=qa.get("ground_truth"),
        answer_text=qa.get("answer"),
        target_step_id=qa.get("target_step_id", []),
        timestamp=qa.get("time"),
    )


def split_samples(
    samples: List[ConversationSample],
    eval_ratio: float,
    seed: int,
) -> tuple[list[ConversationSample], list[ConversationSample]]:
    """Return (eval_split, validation_split) lists."""

    if not samples:
        return [], []
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    cutoff = max(1, min(len(shuffled) - 1, int(len(shuffled) * eval_ratio)))
    eval_split = shuffled[:cutoff]
    validation_split = shuffled[cutoff:]
    return eval_split, validation_split


def dump_jsonl(samples: Iterable[ConversationSample], path: Path) -> None:
    """Write ConversationSample objects to JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.__dict__, ensure_ascii=False) + "\n")


def load_jsonl(
    path: Path,
    *,
    limit: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
    as_dataclass: bool = False,
) -> list:
    """Load a JSONL dataset file."""

    samples: list = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            samples.append(json.loads(line))

    if shuffle:
        random.Random(seed).shuffle(samples)
    if limit:
        samples = samples[:limit]

    if as_dataclass:
        return [ConversationSample(**sample) for sample in samples]
    return samples


__all__ = [
    "ConversationSample",
    "dump_jsonl",
    "iter_conversations",
    "load_jsonl",
    "normalize_messages",
    "split_samples",
]
