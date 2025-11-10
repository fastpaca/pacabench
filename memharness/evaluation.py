from __future__ import annotations

import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Iterable, List

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from .answerers import Answerer
from .types import AnswerResult, ConversationSample, QAInput


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _build_qa_input(sample: ConversationSample) -> QAInput:
    return QAInput(
        sample_id=sample.id,
        history=sample.conversation,
        question=sample.question,
        choices=sample.choices,
        timestamp=sample.timestamp,
        metadata={
            "source_file": sample.source_file,
            "category_path": sample.category_path,
            "target_step_id": sample.target_step_id,
            "tid": sample.tid,
        },
    )


def evaluate_samples(
    samples: Iterable[ConversationSample],
    answerer: Answerer,
    artifact_dir: Path,
    *,
    limit: int | None = None,
    shuffle: bool = False,
    shuffle_seed: int = 42,
) -> dict:
    """Run evaluation over ConversationSample objects with the provided answerer."""

    artifact_dir.mkdir(parents=True, exist_ok=True)
    results_path = artifact_dir / "results.jsonl"

    sample_list = list(samples)
    if shuffle:
        random.Random(shuffle_seed).shuffle(sample_list)
    if limit:
        sample_list = sample_list[:limit]

    total = 0
    correct = 0
    latency_values: List[float] = []
    input_tokens: List[int] = []
    output_tokens: List[int] = []
    total_tokens: List[int] = []
    request_counts: List[int] = []

    progress_iter = sample_list
    if tqdm is not None:
        progress_iter = tqdm(sample_list, desc="Evaluating", unit="sample")

    start_time = time.perf_counter()
    with results_path.open("w", encoding="utf-8") as handle:
        for sample in progress_iter:
            total += 1
            qa_input = _build_qa_input(sample)
            record = {"id": sample.id, "ground_truth": sample.ground_truth}
            sample_start = time.perf_counter()
            try:
                result: AnswerResult = answerer.answer(qa_input)
                latency = time.perf_counter() - sample_start
                record["latency_seconds"] = latency
                latency_values.append(latency)
                record["prediction"] = result.choice
                record["raw_response"] = result.raw_response
                record["success"] = result.choice is not None
                if result.usage:
                    record["input_tokens"] = result.usage.input_tokens
                    record["output_tokens"] = result.usage.output_tokens
                    record["total_tokens"] = result.usage.total_tokens
                    record["requests"] = result.usage.requests
                    input_tokens.append(result.usage.input_tokens)
                    output_tokens.append(result.usage.output_tokens)
                    total_tokens.append(result.usage.total_tokens)
                    request_counts.append(result.usage.requests)
                if result.extra:
                    record["extra"] = result.extra
            except Exception as exc:  # pylint: disable=broad-except
                latency = time.perf_counter() - sample_start
                record["latency_seconds"] = latency
                latency_values.append(latency)
                record["prediction"] = None
                record["success"] = False
                record["error"] = str(exc)

            if record.get("prediction") and record["prediction"] == sample.ground_truth:
                correct += 1

            record["conversation_tokens"] = len(sample.conversation)
            handle.write(json.dumps(record) + "\n")

    elapsed = time.perf_counter() - start_time
    accuracy = correct / total if total else 0.0
    metrics = {
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "elapsed_seconds": elapsed,
        "results_path": str(results_path),
        "avg_latency_seconds": statistics.mean(latency_values) if latency_values else 0.0,
        "p95_latency_seconds": _percentile(latency_values, 0.95),
        "avg_total_tokens": statistics.mean(total_tokens) if total_tokens else 0.0,
        "p95_total_tokens": _percentile(total_tokens, 0.95),
        "avg_input_tokens": statistics.mean(input_tokens) if input_tokens else 0.0,
        "avg_output_tokens": statistics.mean(output_tokens) if output_tokens else 0.0,
        "p95_input_tokens": _percentile(input_tokens, 0.95),
        "p95_output_tokens": _percentile(output_tokens, 0.95),
        "avg_requests": statistics.mean(request_counts) if request_counts else 0.0,
    }
    with (artifact_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


__all__ = ["evaluate_samples"]
