"""Evaluation loop and metrics computation."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics."""

    # Core metrics
    accuracy: float
    total_samples: int
    correct: int
    incorrect: int

    # Token stats
    total_input_tokens: int
    total_output_tokens: int
    avg_input_tokens: float
    avg_output_tokens: float
    median_input_tokens: float
    median_output_tokens: float

    # Latency stats (milliseconds)
    avg_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float

    # Custom aggregated metrics (varies per answerer/dataset)
    custom_metrics: dict[str, Any] = field(default_factory=dict)


async def evaluate_async(
    dataset: Any,
    answerer: Any,
    output_dir: Path | str | None = None,
    limit: int | None = None,
    concurrency: int = 10,
) -> EvalMetrics:
    """Run async evaluation on a dataset with an answerer.

    Args:
        dataset: Dataset instance with load() and evaluate() methods
        answerer: Answerer instance with answer() method
        output_dir: Directory to save results (auto-generated if None)
        limit: Optional limit on number of samples to evaluate
        concurrency: Maximum number of concurrent evaluations

    Returns:
        EvalMetrics with aggregated results
    """
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dataset_name = dataset.__class__.__name__.replace("Dataset", "").lower()
        answerer_name = answerer.__class__.__name__.replace("Answerer", "").lower()
        output_dir = Path("runs") / f"{dataset_name}-{answerer_name}-{timestamp}"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load samples
    print(f"Loading dataset: {dataset.__class__.__name__}")
    samples = dataset.load(limit=limit)
    print(f"Loaded {len(samples)} samples")

    # Save config
    config = {
        "dataset": dataset.__class__.__name__,
        "answerer": answerer.__class__.__name__,
        "num_samples": len(samples),
        "concurrency": concurrency,
        "timestamp": datetime.now().isoformat(),
    }
    with (output_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    # Run evaluation with bounded concurrency
    print(f"Running evaluation (concurrency={concurrency})...")
    eval_results = []
    results_file = output_dir / "results.jsonl"

    # Semaphore for bounded concurrency
    semaphore = asyncio.Semaphore(concurrency)

    # Lock for thread-safe file writing
    file_lock = asyncio.Lock()

    async def evaluate_sample(sample: Any, pbar: tqdm) -> None:
        """Evaluate a single sample with concurrency control."""
        async with semaphore:
            try:
                # Get answer from answerer
                answer_result = await answerer.answer(sample)

                # Evaluate with dataset-specific logic
                eval_result = dataset.evaluate(sample, answer_result)

                # Save incrementally (thread-safe)
                async with file_lock:
                    with results_file.open("a") as f:
                        f.write(json.dumps(asdict(eval_result)) + "\n")

                eval_results.append(eval_result)

            except Exception as e:
                print(f"\nError evaluating sample {sample.id}: {e}")
                # Continue with other samples
            finally:
                pbar.update(1)

    # Create progress bar
    with tqdm(total=len(samples), desc="Evaluating") as pbar:
        # Run all evaluations concurrently (bounded by semaphore)
        tasks = [evaluate_sample(sample, pbar) for sample in samples]
        await asyncio.gather(*tasks)

    # Compute aggregated metrics
    print("Computing metrics...")
    metrics = compute_metrics(eval_results)

    # Save metrics
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(asdict(metrics), f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"Accuracy: {metrics.accuracy:.2%}")
    print(f"Correct: {metrics.correct}/{metrics.total_samples}")
    print(f"Avg Latency: {metrics.avg_latency_ms:.0f}ms")
    print(f"Avg Input Tokens: {metrics.avg_input_tokens:.0f}")
    print(f"Avg Output Tokens: {metrics.avg_output_tokens:.0f}")

    return metrics


def evaluate(
    dataset: Any,
    answerer: Any,
    output_dir: Path | str | None = None,
    limit: int | None = None,
    concurrency: int = 10,
) -> EvalMetrics:
    """Run evaluation on a dataset with an answerer (sync wrapper).

    Args:
        dataset: Dataset instance with load() and evaluate() methods
        answerer: Answerer instance with answer() method
        output_dir: Directory to save results (auto-generated if None)
        limit: Optional limit on number of samples to evaluate
        concurrency: Maximum number of concurrent evaluations

    Returns:
        EvalMetrics with aggregated results
    """
    return asyncio.run(
        evaluate_async(dataset, answerer, output_dir, limit, concurrency)
    )


def compute_metrics(eval_results: list[Any]) -> EvalMetrics:
    """Compute aggregated metrics from evaluation results.

    Args:
        eval_results: List of EvalResult objects

    Returns:
        EvalMetrics with aggregated statistics
    """
    if not eval_results:
        return EvalMetrics(
            accuracy=0.0,
            total_samples=0,
            correct=0,
            incorrect=0,
            total_input_tokens=0,
            total_output_tokens=0,
            avg_input_tokens=0.0,
            avg_output_tokens=0.0,
            median_input_tokens=0.0,
            median_output_tokens=0.0,
            avg_latency_ms=0.0,
            median_latency_ms=0.0,
            p95_latency_ms=0.0,
            p99_latency_ms=0.0,
            min_latency_ms=0.0,
            max_latency_ms=0.0,
        )

    # Extract arrays for statistics
    correctness = [r.correct for r in eval_results]
    latencies = [r.latency_ms for r in eval_results]
    input_tokens = [r.input_tokens for r in eval_results]
    output_tokens = [r.output_tokens for r in eval_results]

    # Core metrics
    total_samples = len(eval_results)
    correct = sum(correctness)
    incorrect = total_samples - correct
    accuracy = correct / total_samples if total_samples > 0 else 0.0

    # Token stats
    total_input_tokens = sum(input_tokens)
    total_output_tokens = sum(output_tokens)
    avg_input_tokens = np.mean(input_tokens)
    avg_output_tokens = np.mean(output_tokens)
    median_input_tokens = np.median(input_tokens)
    median_output_tokens = np.median(output_tokens)

    # Latency stats
    avg_latency_ms = np.mean(latencies)
    median_latency_ms = np.median(latencies)
    p95_latency_ms = np.percentile(latencies, 95)
    p99_latency_ms = np.percentile(latencies, 99)
    min_latency_ms = np.min(latencies)
    max_latency_ms = np.max(latencies)

    # Aggregate custom metrics from results
    custom_metrics = aggregate_custom_metrics(eval_results)

    return EvalMetrics(
        accuracy=accuracy,
        total_samples=total_samples,
        correct=correct,
        incorrect=incorrect,
        total_input_tokens=int(total_input_tokens),
        total_output_tokens=int(total_output_tokens),
        avg_input_tokens=float(avg_input_tokens),
        avg_output_tokens=float(avg_output_tokens),
        median_input_tokens=float(median_input_tokens),
        median_output_tokens=float(median_output_tokens),
        avg_latency_ms=float(avg_latency_ms),
        median_latency_ms=float(median_latency_ms),
        p95_latency_ms=float(p95_latency_ms),
        p99_latency_ms=float(p99_latency_ms),
        min_latency_ms=float(min_latency_ms),
        max_latency_ms=float(max_latency_ms),
        custom_metrics=custom_metrics,
    )


def aggregate_custom_metrics(eval_results: list[Any]) -> dict[str, Any]:
    """Aggregate custom metrics from evaluation results.

    Args:
        eval_results: List of EvalResult objects

    Returns:
        Dictionary of aggregated custom metrics
    """
    # Collect all metric keys
    all_metrics = defaultdict(list)

    for result in eval_results:
        if hasattr(result, "metrics") and result.metrics:
            for key, value in result.metrics.items():
                if isinstance(value, (int, float)):
                    all_metrics[key].append(value)

    # Compute aggregates
    aggregated = {}
    for key, values in all_metrics.items():
        if values:
            aggregated[f"avg_{key}"] = float(np.mean(values))
            aggregated[f"median_{key}"] = float(np.median(values))
            aggregated[f"min_{key}"] = float(np.min(values))
            aggregated[f"max_{key}"] = float(np.max(values))

    return aggregated
