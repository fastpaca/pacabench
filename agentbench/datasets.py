"""Dataset loaders for AgentBench."""

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download


@dataclass
class Case:
    """A single test case."""

    id: str
    task_type: str
    inputs: dict[str, Any]
    expected_output: str
    metadata: dict[str, Any]


def load_membench(
    agent_type: str = "FirstAgent",
    split: str = "eval",
    eval_ratio: float = 0.8,
    seed: int = 42,
    limit: int | None = None,
    data_dir: Path | str | None = None,
) -> list[Case]:
    """
    Load MemBench dataset.

    Args:
        agent_type: "FirstAgent" or "ThirdAgent"
        split: "eval", "validation", or "all"
        eval_ratio: Ratio for eval/validation split
        seed: Random seed for split
        limit: Optional limit on number of samples
        data_dir: Path to MemBench data directory

    Returns:
        List of Cases
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "MemBench"
    data_dir = Path(data_dir)
    source_dir = data_dir / "MemData" / agent_type

    if not source_dir.exists():
        raise FileNotFoundError(f"MemBench data not found at {source_dir}")

    all_samples = list(_iter_membench_samples(source_dir))
    all_samples.sort(key=lambda s: s.id)

    if split == "all":
        samples = all_samples
    else:
        eval_samples, val_samples = _split_samples(all_samples, eval_ratio, seed)
        samples = eval_samples if split == "eval" else val_samples

    if limit is not None:
        samples = samples[:limit]

    return [
        Case(
            id=sample.id,
            task_type="qa",
            inputs={
                "conversation": sample.conversation,
                "question": sample.question,
                "choices": sample.choices,
            },
            expected_output=sample.ground_truth,
            metadata=sample.metadata,
        )
        for sample in samples
    ]


def load_longmemeval(
    split: str = "s_cleaned",
    limit: int | None = None,
    cache_dir: Path | str | None = None,
) -> list[Case]:
    """
    Load LongMemEval dataset.

    Args:
        split: "s_cleaned" (single-session) or "m_cleaned" (multi-session)
        limit: Optional limit on number of samples
        cache_dir: Optional cache directory for HuggingFace downloads

    Returns:
        List of Cases
    """
    split_files = {
        "s_cleaned": "longmemeval_s_cleaned.json",
        "m_cleaned": "longmemeval_m_cleaned.json",
    }

    if split not in split_files:
        raise ValueError(f"Invalid split '{split}'. Available: {list(split_files.keys())}")

    file_path = hf_hub_download(
        repo_id="xiaowu0162/longmemeval-cleaned",
        filename=split_files[split],
        repo_type="dataset",
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    with open(file_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    if limit is not None:
        raw_data = raw_data[:limit]

    cases = []
    for sample in raw_data:
        conversation = _flatten_sessions(sample["haystack_sessions"])
        cases.append(
            Case(
                id=sample["question_id"],
                task_type="qa",
                inputs={
                    "conversation": conversation,
                    "question": sample["question"],
                },
                expected_output=sample["answer"],
                metadata={
                    "question_type": sample["question_type"],
                    "question_date": sample["question_date"],
                    "answer_session_ids": sample["answer_session_ids"],
                },
            )
        )

    return cases


def load_gaia(
    level: str = "all",
    split: str = "validation",
    limit: int | None = None,
) -> list[Case]:
    """
    Load GAIA dataset.

    Args:
        level: Difficulty level ("level1", "level2", "level3", or "all")
        split: Dataset split ("validation" or "test")
        limit: Optional limit on number of samples

    Returns:
        List of Cases
    """
    levels = {
        "level1": "2023_level1",
        "level2": "2023_level2",
        "level3": "2023_level3",
        "all": "2023_all",
    }

    if level not in levels:
        raise ValueError(f"Invalid level '{level}'. Available: {list(levels.keys())}")

    dataset_name = levels[level]
    hf_dataset = load_dataset("gaia-benchmark/GAIA", dataset_name, split=split)

    if limit is not None:
        hf_dataset = hf_dataset.select(range(min(limit, len(hf_dataset))))

    cases = []
    for item in hf_dataset:
        inputs: dict[str, Any] = {
            "question": item["Question"],
        }

        if item.get("file_name"):
            inputs["file_name"] = item["file_name"]
            inputs["file_path"] = item.get("file_path", "")

        if item.get("Annotator Metadata"):
            inputs["metadata"] = item["Annotator Metadata"]

        cases.append(
            Case(
                id=item.get("task_id", f"gaia_{len(cases)}"),
                task_type="agentic",
                inputs=inputs,
                expected_output=item["Final answer"],
                metadata={
                    "level": item.get("Level", ""),
                    "file_name": item.get("file_name", ""),
                },
            )
        )

    return cases


@dataclass
class _MemBenchSample:
    """Internal sample representation for MemBench."""

    id: str
    conversation: list[dict[str, str]]
    question: str
    choices: dict[str, str]
    ground_truth: str
    metadata: dict[str, Any]


def _iter_membench_samples(source_dir: Path) -> Iterator[_MemBenchSample]:
    """Iterate through all JSON files and extract samples."""
    for json_path in sorted(source_dir.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        yield from _extract_membench_samples(payload, json_path.stem)


def _extract_membench_samples(
    node: Any,
    dataset_name: str,
    category_path: list[str] | None = None,
) -> Iterator[_MemBenchSample]:
    """Recursively extract samples from nested JSON structure."""
    if category_path is None:
        category_path = []

    if isinstance(node, list):
        for item in node:
            if isinstance(item, dict) and "message_list" in item and "QA" in item:
                yield _build_membench_sample(item, dataset_name, category_path)
            else:
                yield from _extract_membench_samples(item, dataset_name, category_path)
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from _extract_membench_samples(value, dataset_name, category_path + [key])


def _build_membench_sample(
    record: dict,
    dataset_name: str,
    category_path: list[str],
) -> _MemBenchSample:
    """Build a sample from a JSON record."""
    qa = record.get("QA", {})
    conversation = _normalize_membench_messages(record.get("message_list", []))
    sample_id = f"{dataset_name}::{record.get('tid', 'unknown')}"

    return _MemBenchSample(
        id=sample_id,
        conversation=conversation,
        question=qa.get("question", ""),
        choices=qa.get("choices", {}),
        ground_truth=qa.get("ground_truth", ""),
        metadata={
            "source_file": dataset_name,
            "category_path": category_path,
            "tid": record.get("tid"),
        },
    )


def _normalize_membench_messages(message_list: list) -> list[dict[str, str]]:
    """Flatten nested message structure into simple turn dictionaries."""
    turns = []

    for block in message_list:
        sessions = block if isinstance(block, list) else [block]
        for message in sessions:
            if not isinstance(message, dict):
                turns.append({"role": "user", "content": str(message)})
                continue

            time_val = message.get("time")
            place_val = message.get("place")

            meta_parts = []
            if time_val:
                meta_parts.append(f"time: {time_val}")
            if place_val:
                meta_parts.append(f"place: {place_val}")
            meta_suffix = f" ({'; '.join(meta_parts)})" if meta_parts else ""

            user_text = message.get("user_message") or message.get("user")
            assistant_text = message.get("assistant_message") or message.get("assistant")

            if user_text:
                turns.append({"role": "user", "content": f"{user_text}{meta_suffix}"})
            if assistant_text:
                turns.append({"role": "assistant", "content": f"{assistant_text}{meta_suffix}"})

    return [turn for turn in turns if turn.get("content")]


def _flatten_sessions(sessions: list[list[dict[str, str]]]) -> list[dict[str, str]]:
    """Flatten nested session structure into a single conversation history."""
    turns = []

    for session in sessions:
        if isinstance(session, list):
            turns.extend(session)
        elif isinstance(session, dict):
            turns.append(session)

    return [turn for turn in turns if isinstance(turn, dict) and turn.get("content")]


def _split_samples(
    samples: list[Any],
    eval_ratio: float,
    seed: int,
) -> tuple[list[Any], list[Any]]:
    """Split samples into eval and validation sets."""
    if not samples:
        return [], []

    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)

    cutoff = max(1, min(len(shuffled) - 1, int(len(shuffled) * eval_ratio)))
    eval_split = shuffled[:cutoff]
    validation_split = shuffled[cutoff:]

    return eval_split, validation_split
