import os
import time

from openai import AsyncOpenAI

from agentbench.evaluators.base import BaseEvaluator
from agentbench.types import Case, EvaluationResult, RunnerOutput


class LLMJudgeEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = self.config.model or "gpt-4o-mini"

    async def evaluate(self, case: Case, output: RunnerOutput) -> EvaluationResult:
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

        response_text = output.output.strip()
        expected_text = expected.strip()

        # Construct prompt
        # We assume case.input is the question/prompt
        question = case.input

        prompt = f"""You are evaluating if a model's answer is semantically equivalent to the expected answer.

Question: {question}

Expected Answer: {expected_text}

Model's Answer: {response_text}

Does the model's answer convey the same information as the expected answer? Consider:
- Paraphrasing is acceptable
- Extra explanation is acceptable if core answer is present
- Minor details can differ if main point matches

Respond with ONLY "YES" or "NO"."""

        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )

            judgment = completion.choices[0].message.content or ""
            judgment = judgment.strip().upper()
            passed = judgment.startswith("YES")

            usage = {
                "input_tokens": completion.usage.prompt_tokens if completion.usage else 0,
                "output_tokens": completion.usage.completion_tokens if completion.usage else 0,
            }

            return EvaluationResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                reason=judgment,
                evaluator_latency_ms=(time.perf_counter() - start) * 1000,
                metrics=usage,
            )

        except Exception as e:
            return EvaluationResult(
                passed=False,
                score=0.0,
                reason=f"Judge error: {str(e)}",
                evaluator_latency_ms=(time.perf_counter() - start) * 1000,
            )
