"""Mem0 QA runner with memory retrieval."""

import time

from mem0 import AsyncMemory
from pydantic_ai import Agent, ModelRequest, UserPromptPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from agentbench.types import Case, Runner, RunnerContext, RunnerOutput


class Mem0Runner(Runner):
    """Mem0 QA runner with memory retrieval."""

    async def run_case(self, case: Case, ctx: RunnerContext) -> RunnerOutput:
        """
        Run a QA case with Mem0 memory retrieval.

        Args:
            case: Test case
            ctx: Evaluation context

        Returns:
            RunnerOutput with output, error, and metrics
        """
        start_time = time.time()

        try:
            vector_store = {
                "provider": "qdrant",
                "config": {"collection_name": "agentbench", "host": "memory"},
            }
            embedder_config = {"provider": "openai"}
            if ctx.embedding_model:
                embedder_config["config"] = {"model": ctx.embedding_model}

            config = {"vector_store": vector_store, "embedder": embedder_config}

            memory = await AsyncMemory.from_config(config)
            model_obj = OpenAIChatModel(
                ctx.model,
                provider=OpenAIProvider(
                    base_url=f"http://localhost:{ctx.proxy_port}/v1",
                    api_key=ctx.openai_api_key,
                ),
            )
            agent = Agent(model=model_obj)

            user_id = case.id
            conversation = case.inputs["conversation"]
            question = case.inputs["question"]
            choices = case.inputs.get("choices")

            if conversation:
                messages = [
                    {"role": msg["role"], "content": msg["content"]} for msg in conversation
                ]
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
                    memory_context = (
                        "Relevant context from memory:\n" + "\n".join(memory_lines) + "\n\n"
                    )

            message_history = [
                ModelRequest(parts=[UserPromptPart(content=memory_context)])
                if memory_context
                else None
            ]
            message_history = [m for m in message_history if m]

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
