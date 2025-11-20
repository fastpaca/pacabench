import os
import time

from genai_prices import Usage, calc_price
from openai import AsyncOpenAI

from agentbench.evaluators.base import BaseEvaluator
from agentbench.types import Case, EvaluationResult, RunnerOutput


class LLMJudgeEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self.model = self.config.model or "gpt-4o-mini"
        self._default_base_url = os.getenv("OPENAI_BASE_URL")
        self._api_key = os.getenv("OPENAI_API_KEY")

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

        response_text = output.output.strip()
        expected_text = expected.strip()
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

        client = self._build_client(proxy_url)

        try:
            completion = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )

            judgment = completion.choices[0].message.content or ""
            judgment = judgment.strip().upper()
            passed = judgment.startswith("YES")

            input_tokens = completion.usage.prompt_tokens if completion.usage else 0
            output_tokens = completion.usage.completion_tokens if completion.usage else 0

            # Calculate cost
            cost_usd = 0.0
            try:
                usage_obj = Usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                price_calc = calc_price(usage_obj, self.model)
                cost_usd = float(price_calc.total_price)
            except Exception:
                # If cost calculation fails (e.g. unknown model), default to 0
                pass

            usage = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
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

    def _build_client(self, proxy_url: str | None) -> AsyncOpenAI:
        # NOTE: We explicitly prefer direct OpenAI connection (or env var) for the judge,
        # unless the user forces everything through proxy.
        # But the spec says: "we don't need to route the judge eval to the proxy"
        # So we will ignore proxy_url if we have a direct key/url, or use default.
        # Actually, to be safe and independent, we should just use standard env vars.

        # However, the current code passed proxy_url. We should probably ignore it
        # if we want to enforce "side effect cost".
        # But if the user runs locally with no internet and a local LLM at proxy_url,
        # we might want to use it?
        # The user instruction: "we don't need to route the judge eval to the proxy but we should include it as a secondary cost metric"
        # So using direct OpenAI is fine.

        base_url = self._default_base_url or "https://api.openai.com/v1"
        return AsyncOpenAI(api_key=self._api_key, base_url=base_url)
