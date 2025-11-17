"""Long-context QA runner.

Baseline approach: Pass full conversation history to LLM context window.
"""

import time

from pydantic_ai import Agent, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from agentbench.types import Case, Runner, RunnerContext, RunnerOutput


class LongContextRunner(Runner):
    """Long-context QA runner that passes full conversation history to LLM."""

    async def run_case(self, case: Case, ctx: RunnerContext) -> RunnerOutput:
        """
        Run a QA case with long context.

        Args:
            case: Test case
            ctx: Evaluation context

        Returns:
            RunnerOutput with output, error, and metrics
        """
        start_time = time.time()

        try:
            model_obj = OpenAIChatModel(
                ctx.model,
                provider=OpenAIProvider(
                    base_url=f"http://localhost:{ctx.proxy_port}/v1",
                    api_key=ctx.openai_api_key,
                ),
            )

            agent = Agent(model=model_obj)

            conversation = case.inputs["conversation"]
            question = case.inputs["question"]
            choices = case.inputs.get("choices")

            message_history = []
            for turn in conversation:
                if turn["role"] == "user":
                    message_history.append(
                        ModelRequest(parts=[UserPromptPart(content=turn["content"])])
                    )
                elif turn["role"] == "assistant":
                    message_history.append(ModelResponse(parts=[TextPart(content=turn["content"])]))

            prompt = f"Question: {question}"

            if choices:
                choices_text = "\n".join(
                    f"{key}. {value}" for key, value in sorted(choices.items())
                )
                prompt += f"\n\nChoices:\n{choices_text}\n\nRespond with only the choice letter (A, B, C, or D)."

            result = await agent.run(prompt, message_history=message_history)
            output = str(result.output)

            duration_ms = (time.time() - start_time) * 1000
            return RunnerOutput(output=output, error=None, duration_ms=duration_ms)
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return RunnerOutput(
                output=None,
                error=f"Runner execution failed: {e}",
                duration_ms=duration_ms,
            )
