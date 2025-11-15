#!/usr/bin/env python3
import asyncio
import json
import os
import sys

from mem0 import AsyncMemory
from pydantic_ai import Agent, ModelRequest, UserPromptPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


async def main_async():
    case = json.loads(sys.stdin.read())
    model = os.getenv("MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    embedding_model = os.getenv("EMBEDDING_MODEL")

    vector_store = {
        "provider": "qdrant",
        "config": {"collection_name": "agentbench", "host": "memory"},
    }
    embedder_config = {"provider": "openai"}
    if embedding_model:
        embedder_config["config"] = {"model": embedding_model}

    config = {"vector_store": vector_store, "embedder": embedder_config}

    memory = await AsyncMemory.from_config(config)
    model_obj = OpenAIChatModel(model, provider=OpenAIProvider(base_url=base_url, api_key=api_key))
    agent = Agent(model=model_obj)

    user_id = case["id"]
    conversation = case["inputs"]["conversation"]
    question = case["inputs"]["question"]
    choices = case["inputs"].get("choices")

    if conversation:
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in conversation]
        await memory.add(messages, user_id=user_id)

    search_result = await memory.search(query=question, user_id=user_id, limit=5)
    relevant_memories = (
        search_result.get("results", [])
        if isinstance(search_result, dict)
        else (search_result if isinstance(search_result, list) else [])
    )

    memory_context = ""
    if relevant_memories:
        memory_lines = []
        for idx, mem in enumerate(relevant_memories, 1):
            memory_text = (
                mem.get("memory", "") or mem.get("text", "")
                if isinstance(mem, dict)
                else (mem if isinstance(mem, str) else str(mem))
            )
            if memory_text:
                memory_lines.append(f"{idx}. {memory_text}")
        if memory_lines:
            memory_context = "Relevant context from memory:\n" + "\n".join(memory_lines) + "\n\n"

    message_history = [
        ModelRequest(parts=[UserPromptPart(content=memory_context)]) if memory_context else None
    ]
    message_history = [m for m in message_history if m]

    prompt = f"Question: {question}"
    if choices:
        choices_text = "\n".join(f"{key}. {value}" for key, value in sorted(choices.items()))
        prompt += (
            f"\n\nChoices:\n{choices_text}\n\nRespond with only the choice letter (A, B, C, or D)."
        )

    try:
        result = await agent.run(prompt, message_history=message_history)
        print(json.dumps({"result": str(result.output), "error": None}))
    except Exception as e:
        print(json.dumps({"result": None, "error": str(e)}))
        sys.exit(1)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
