#!/usr/bin/env python3
"""
Long-context QA runner.

Baseline approach: Pass full conversation history to LLM context window.
Reads test case from stdin, outputs result to stdout.
"""

import asyncio
import json
import os
import sys

from pydantic_ai import Agent, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


async def main_async() -> None:
    """Main runner entry point (async)."""
    case = json.loads(sys.stdin.read())

    model = os.getenv("MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")

    model_obj = OpenAIChatModel(
        model,
        provider=OpenAIProvider(
            base_url=base_url,
            api_key=api_key,
        ),
    )

    agent = Agent(model=model_obj)

    conversation = case["inputs"]["conversation"]
    question = case["inputs"]["question"]
    choices = case["inputs"].get("choices")

    message_history = []
    for turn in conversation:
        if turn["role"] == "user":
            message_history.append(ModelRequest(parts=[UserPromptPart(content=turn["content"])]))
        elif turn["role"] == "assistant":
            message_history.append(ModelResponse(parts=[TextPart(content=turn["content"])]))

    prompt = f"Question: {question}"

    if choices:
        choices_text = "\n".join(f"{key}. {value}" for key, value in sorted(choices.items()))
        prompt += (
            f"\n\nChoices:\n{choices_text}\n\nRespond with only the choice letter (A, B, C, or D)."
        )

    try:
        result = await agent.run(prompt, message_history=message_history)
        output = str(result.output)

        print(json.dumps({"result": output, "error": None}))
    except Exception as e:
        print(json.dumps({"result": None, "error": str(e)}))
        sys.exit(1)


def main() -> None:
    """Sync wrapper."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
