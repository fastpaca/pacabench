"""MemBench dataset implementation."""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext


class MultipleChoiceEvaluator(Evaluator):
    """Evaluator for multiple choice questions."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        """Check if response choice matches expected."""
        response = ctx.output.strip().upper() if ctx.output else ""
        choice = response[0] if response else ""
        expected = ctx.expected_output.strip().upper() if ctx.expected_output else ""
        return choice == expected


def load_membench(
    agent_type: str = "FirstAgent",
    split: str = "eval",
    eval_ratio: float = 0.8,
    seed: int = 42,
    limit: int | None = None,
    data_dir: Path | str | None = None,
) -> Dataset:
    """Load MemBench dataset as pydantic-evals Dataset.

    Args:
        agent_type: "FirstAgent" or "ThirdAgent"
        split: "eval", "validation", or "all"
        eval_ratio: Ratio for eval/validation split
        seed: Random seed for split
        limit: Optional limit on number of samples
        data_dir: Path to MemBench data directory

    Returns:
        pydantic-evals Dataset with Cases
    """
    # Setup paths
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data" / "MemBench"
    data_dir = Path(data_dir)
    source_dir = data_dir / "MemData" / agent_type

    if not source_dir.exists():
        raise FileNotFoundError(
            f"MemBench data not found at {source_dir}. Please ensure data is available."
        )

    # Load all samples
    all_samples = list(_iter_samples(source_dir))
    all_samples.sort(key=lambda s: s.id)

    # Apply split
    if split == "all":
        samples = all_samples
    else:
        eval_samples, val_samples = _split_samples(all_samples, eval_ratio, seed)
        samples = eval_samples if split == "eval" else val_samples

    # Apply limit
    if limit is not None:
        samples = samples[:limit]

    # Convert to pydantic-evals Cases
    cases = [
        Case(
            name=sample.id,
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

    return Dataset(cases=cases, evaluators=[MultipleChoiceEvaluator()])


# Internal types and helpers


@dataclass
class _MemBenchSample:
    """Internal sample representation."""

    id: str
    conversation: list[dict[str, str]]
    question: str
    choices: dict[str, str]
    ground_truth: str
    metadata: dict[str, Any]


def _iter_samples(source_dir: Path) -> Iterator[_MemBenchSample]:
    """Iterate through all JSON files and extract samples."""
    for json_path in sorted(source_dir.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        yield from _extract_samples(payload, json_path.stem)


def _extract_samples(
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
                yield _build_sample(item, dataset_name, category_path)
            else:
                yield from _extract_samples(item, dataset_name, category_path)
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from _extract_samples(value, dataset_name, category_path + [key])


def _build_sample(
    record: dict,
    dataset_name: str,
    category_path: list[str],
) -> _MemBenchSample:
    """Build a sample from a JSON record."""
    qa = record.get("QA", {})
    conversation = _normalize_messages(record.get("message_list", []))
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
            "target_step_id": qa.get("target_step_id", []),
            "timestamp": qa.get("time"),
            "answer_text": qa.get("answer"),
        },
    )


def _normalize_messages(message_list: list) -> list[dict[str, str]]:
    """Flatten nested message structure into simple turn dictionaries."""
    turns = []

    for block in message_list:
        sessions = block if isinstance(block, list) else [block]
        for message in sessions:
            if not isinstance(message, dict):
                turns.append({"role": "user", "content": str(message)})
                continue

            # Extract metadata
            time_val = message.get("time")
            place_val = message.get("place")

            # Format metadata suffix
            meta_parts = []
            if time_val:
                meta_parts.append(f"time: {time_val}")
            if place_val:
                meta_parts.append(f"place: {place_val}")
            meta_suffix = f" ({'; '.join(meta_parts)})" if meta_parts else ""

            # Extract messages
            user_text = message.get("user_message") or message.get("user")
            assistant_text = message.get("assistant_message") or message.get("assistant")

            if user_text:
                turns.append({"role": "user", "content": f"{user_text}{meta_suffix}"})
            if assistant_text:
                turns.append({"role": "assistant", "content": f"{assistant_text}{meta_suffix}"})

    return [turn for turn in turns if turn.get("content")]


def _split_samples(
    samples: list[_MemBenchSample],
    eval_ratio: float,
    seed: int,
) -> tuple[list[_MemBenchSample], list[_MemBenchSample]]:
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
