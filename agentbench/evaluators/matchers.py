import re
import time
from typing import Any

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

        threshold = self.config.extra_config.get("threshold", 0.5)
        passed = f1 >= threshold

        return EvaluationResult(
            passed=passed,
            score=f1,
            reason=f"F1 Score: {f1:.2f}",
            evaluator_latency_ms=(time.perf_counter() - start) * 1000,
        )


def _normalize_choice_map(choices: Any) -> dict[str, str]:
    if not isinstance(choices, dict):
        return {}
    normalized: dict[str, str] = {}
    for key, value in choices.items():
        key_str = str(key).strip().upper()
        if not key_str:
            continue
        normalized[key_str] = str(value).strip()
    return normalized


def _extract_choice_letter(text: str, valid_letters: set[str]) -> str | None:
    tokens = re.findall(r"[A-Za-z]", text.upper())
    for token in tokens:
        if token in valid_letters:
            return token
    return None


def _clean_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return cleaned.strip()


class MultipleChoiceEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        fallback = self.config.extra_config.get("fallback", "f1")
        self._fallback_mode = fallback if fallback in {"f1", "judge", "none"} else "f1"
        self._judge_model = self.config.model or "gpt-4o-mini"

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

        choices = _normalize_choice_map(case.metadata.get("choices"))
        expected_text = (case.expected or "").strip()
        expected_letter = expected_text[0].upper() if expected_text else ""

        # Primary: strict multiple-choice letter check.
        if choices and expected_letter:
            valid_letters = set(choices.keys())
            predicted_letter = _extract_choice_letter(output.output, valid_letters)

            if predicted_letter:
                passed = predicted_letter == expected_letter
                reason = f"pred={predicted_letter}, expected={expected_letter}"
                return EvaluationResult(
                    passed=passed,
                    score=1.0 if passed else 0.0,
                    reason=reason,
                    evaluator_latency_ms=(time.perf_counter() - start) * 1000,
                )

            cleaned_output = _clean_text(output.output)
            for key, value in choices.items():
                if cleaned_output == _clean_text(value):
                    passed = key == expected_letter
                    reason = f"matched_choice_value={key}, expected={expected_letter}"
                    return EvaluationResult(
                        passed=passed,
                        score=1.0 if passed else 0.0,
                        reason=reason,
                        evaluator_latency_ms=(time.perf_counter() - start) * 1000,
                    )

            return EvaluationResult(
                passed=False,
                score=0.0,
                reason="Could not extract choice from output",
                evaluator_latency_ms=(time.perf_counter() - start) * 1000,
            )

        # Fallbacks for cases without choices or expected letter.
        if self._fallback_mode == "judge":
            from agentbench.config import EvaluatorConfig
            from agentbench.evaluators.judge import LLMJudgeEvaluator

            judge = LLMJudgeEvaluator(EvaluatorConfig(type="llm_judge", model=self._judge_model))
            return await judge.evaluate(case, output, proxy_url=proxy_url)

        if self._fallback_mode == "f1":
            f1_eval = F1Evaluator(self.config)
            return await f1_eval.evaluate(case, output, proxy_url=proxy_url)

        return EvaluationResult(
            passed=False,
            score=0.0,
            reason="No choices present and fallback disabled",
            evaluator_latency_ms=(time.perf_counter() - start) * 1000,
        )
