import time

import tiktoken

from agentbench.evaluators.base import BaseEvaluator
from agentbench.types import Case, EvaluationResult, RunnerOutput


class ExactMatchEvaluator(BaseEvaluator):
    async def evaluate(
        self, case: Case, output: RunnerOutput, proxy_url: str | None = None
    ) -> EvaluationResult:
        start = time.perf_counter()
        if output.error or not output.output:
            return EvaluationResult(
                passed=False,
                score=0.0,
                reason="No output or error",
                evaluator_latency_ms=(time.perf_counter() - start) * 1000,
            )

        expected = case.expected
        if not expected:
            # If no expected output, can't evaluate? Or passed?
            return EvaluationResult(
                passed=True,
                score=1.0,
                reason="No expected output provided",
                evaluator_latency_ms=(time.perf_counter() - start) * 1000,
            )

        passed = output.output.strip() == expected.strip()
        return EvaluationResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reason="Exact match" if passed else "Mismatch",
            evaluator_latency_ms=(time.perf_counter() - start) * 1000,
        )


class F1Evaluator(BaseEvaluator):
    async def evaluate(
        self, case: Case, output: RunnerOutput, proxy_url: str | None = None
    ) -> EvaluationResult:
        start = time.perf_counter()
        if output.error or not output.output:
            return EvaluationResult(
                passed=False,
                score=0.0,
                reason="No output or error",
                evaluator_latency_ms=(time.perf_counter() - start) * 1000,
            )

        expected = case.expected
        if not expected:
            return EvaluationResult(
                passed=True,
                score=1.0,
                reason="No expected output provided",
                evaluator_latency_ms=(time.perf_counter() - start) * 1000,
            )

        # Simple token overlap F1
        # Using cl100k_base as default
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            encoding = tiktoken.get_encoding("gpt2")

        pred_tokens = set(encoding.encode(output.output.strip().lower()))
        ref_tokens = set(encoding.encode(expected.strip().lower()))

        if not ref_tokens:
            return EvaluationResult(
                passed=True,
                score=1.0,
                reason="Empty reference",
                evaluator_latency_ms=(time.perf_counter() - start) * 1000,
            )

        common = pred_tokens & ref_tokens
        precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
        recall = len(common) / len(ref_tokens) if ref_tokens else 0.0

        f1 = 0.0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)

        # Pass threshold? usually 0.5? Configurable?
        threshold = self.config.extra_config.get("threshold", 0.5)
        passed = f1 >= threshold

        return EvaluationResult(
            passed=passed,
            score=f1,
            reason=f"F1 Score: {f1:.2f}",
            evaluator_latency_ms=(time.perf_counter() - start) * 1000,
        )
