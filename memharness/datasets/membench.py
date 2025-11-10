"""MemBench dataset implementation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

# Types colocated with dataset
@dataclass
class MemBenchSample:
    """A single MemBench QA sample."""

    id: str
    conversation: list[dict[str, str]]  # List of {role, content}
    question: str
    choices: dict[str, str]  # {A: ..., B: ..., C: ..., D: ...}
    ground_truth: str  # "A", "B", "C", or "D"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result of evaluating a single sample."""

    correct: bool
    sample_id: str
    response: str
    ground_truth: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    metrics: dict[str, Any] = field(default_factory=dict)


class MemBenchDataset:
    """MemBench dataset for memory QA evaluation."""

    def __init__(
        self,
        data_dir: Path | str | None = None,
        agent_type: str = "FirstAgent",
        split: str = "eval",
        eval_ratio: float = 0.8,
        seed: int = 42,
    ):
        """Initialize MemBench dataset.

        Args:
            data_dir: Path to MemBench data directory (defaults to data/MemBench)
            agent_type: "FirstAgent" or "ThirdAgent"
            split: "eval", "validation", or "all"
            eval_ratio: Ratio of samples to use for eval vs validation
            seed: Random seed for split
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data" / "MemBench"
        self.data_dir = Path(data_dir)
        self.agent_type = agent_type
        self.split = split
        self.eval_ratio = eval_ratio
        self.seed = seed

        self.source_dir = self.data_dir / "MemData" / agent_type
        if not self.source_dir.exists():
            raise FileNotFoundError(
                f"MemBench data not found at {self.source_dir}. "
                f"Please ensure data is available or download from HuggingFace."
            )

    def load(self, limit: int | None = None) -> list[MemBenchSample]:
        """Load samples from the dataset.

        Args:
            limit: Optional limit on number of samples to return

        Returns:
            List of MemBenchSample objects
        """
        # Load all samples from JSON files
        all_samples = list(self._iter_samples())

        # Sort by ID for reproducibility
        all_samples.sort(key=lambda s: s.id)

        # Apply split
        if self.split == "all":
            samples = all_samples
        else:
            eval_samples, val_samples = self._split_samples(all_samples)
            samples = eval_samples if self.split == "eval" else val_samples

        # Apply limit
        if limit is not None:
            samples = samples[:limit]

        return samples

    def evaluate(self, sample: MemBenchSample, answer_result: Any) -> EvalResult:
        """Evaluate an answerer's response against ground truth.

        Args:
            sample: The MemBench sample
            answer_result: AnswerResult from answerer

        Returns:
            EvalResult with correctness and metrics
        """
        # Multiple choice evaluation - extract choice letter and compare
        response = answer_result.response.strip().upper()

        # Handle various response formats (just "A", "A.", "A)", etc.)
        if len(response) > 0:
            choice = response[0]
        else:
            choice = ""

        correct = (choice == sample.ground_truth.strip().upper())

        return EvalResult(
            correct=correct,
            sample_id=sample.id,
            response=choice,
            ground_truth=sample.ground_truth,
            latency_ms=answer_result.total_latency_ms,
            input_tokens=answer_result.input_tokens,
            output_tokens=answer_result.output_tokens,
            metrics=answer_result.metrics,
        )

    def _iter_samples(self) -> Iterator[MemBenchSample]:
        """Iterate through all JSON files and extract samples."""
        for json_path in sorted(self.source_dir.glob("*.json")):
            with json_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)

            # Recursively extract samples from nested structure
            yield from self._extract_samples(payload, json_path.stem)

    def _extract_samples(
        self,
        node: Any,
        dataset_name: str,
        category_path: list[str] | None = None,
    ) -> Iterator[MemBenchSample]:
        """Recursively extract samples from nested JSON structure."""
        if category_path is None:
            category_path = []

        if isinstance(node, list):
            for item in node:
                if isinstance(item, dict) and "message_list" in item and "QA" in item:
                    yield self._build_sample(item, dataset_name, category_path)
                else:
                    yield from self._extract_samples(item, dataset_name, category_path)
        elif isinstance(node, dict):
            for key, value in node.items():
                yield from self._extract_samples(
                    value, dataset_name, category_path + [key]
                )

    def _build_sample(
        self,
        record: dict,
        dataset_name: str,
        category_path: list[str],
    ) -> MemBenchSample:
        """Build a MemBenchSample from a JSON record."""
        qa = record.get("QA", {})
        conversation = self._normalize_messages(record.get("message_list", []))

        sample_id = f"{dataset_name}::{record.get('tid', 'unknown')}"

        return MemBenchSample(
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

    def _normalize_messages(self, message_list: list) -> list[dict[str, str]]:
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
                    turns.append({
                        "role": "user",
                        "content": f"{user_text}{meta_suffix}"
                    })
                if assistant_text:
                    turns.append({
                        "role": "assistant",
                        "content": f"{assistant_text}{meta_suffix}"
                    })

        return [turn for turn in turns if turn.get("content")]

    def _split_samples(
        self,
        samples: list[MemBenchSample],
    ) -> tuple[list[MemBenchSample], list[MemBenchSample]]:
        """Split samples into eval and validation sets."""
        if not samples:
            return [], []

        rng = random.Random(self.seed)
        shuffled = samples[:]
        rng.shuffle(shuffled)

        cutoff = max(1, min(len(shuffled) - 1, int(len(shuffled) * self.eval_ratio)))
        eval_split = shuffled[:cutoff]
        validation_split = shuffled[cutoff:]

        return eval_split, validation_split
