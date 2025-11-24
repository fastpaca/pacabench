import json
import os
import sys
from typing import Literal

from mem0 import Memory
from openai import OpenAI
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class InputCase(BaseModel):
    case_id: str
    input: str
    history: list[Message] = Field(default_factory=list)
    choices: dict[str, str] = Field(default_factory=dict)


def main():
    # Configuration
    proxy_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    qdrant_host = os.getenv("MEM0_QDRANT_HOST", "localhost")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Configure Mem0
    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "pacabench",
                "host": qdrant_host,
                "port": 6333,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small",
                "openai_base_url": proxy_url,
                "api_key": api_key,
            },
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": model,
                "openai_base_url": proxy_url,
                "api_key": api_key,
            },
        },
    }

    try:
        m = Memory.from_config(config)
    except Exception as e:
        sys.stderr.write(f"Failed to init Memory with Qdrant: {e}\n")
        sys.exit(1)

    # Initialize OpenAI client for response generation
    client = OpenAI(api_key=api_key, base_url=proxy_url)

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            case = InputCase.model_validate(data)
        except Exception:
            continue

        user_id = str(case.case_id)

        try:
            # 1. Populate memory with history
            # Format history for mem0: list of dicts, but mem0 add expects list of messages or text.
            # m.add(messages, user_id=...)
            history_dicts = [m.model_dump() for m in case.history]
            m.add(history_dicts, user_id=user_id)

            # 2. Search memory
            search_result = m.search(query=case.input, user_id=user_id, limit=5)

            # Mem0 returns list of dicts: [{'memory': '...', 'score': ...}, ...]
            relevant_memories = search_result if isinstance(search_result, list) else []

            memory_context = ""
            if relevant_memories:
                memory_lines = []
                for idx, mem in enumerate(relevant_memories, 1):
                    # mem is a dict
                    if not isinstance(mem, dict):
                        continue
                    memory_text = mem.get("memory") or mem.get("text") or ""
                    if memory_text:
                        memory_lines.append(f"{idx}. {memory_text}")

                if memory_lines:
                    memory_context = (
                        "Relevant context from memory:\n" + "\n".join(memory_lines) + "\n\n"
                    )

            # 3. Generate Response
            prompt_content = f"{memory_context}Question: {case.input}"
            system_prompt = "You are a helpful assistant with a long memory."

            if case.choices:
                choices_text = "\n".join(
                    f"{key}. {value}" for key, value in sorted(case.choices.items())
                )
                prompt_content += f"\n\nChoices:\n{choices_text}\n\nRespond with only the choice letter (A, B, C, or D)."

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_content},
                ],
            )
            output = response.choices[0].message.content

            print(json.dumps({"output": output, "error": None}))
        except Exception as e:
            sys.stderr.write(f"Error in agent loop: {e}\n")
            print(json.dumps({"output": None, "error": str(e)}))

        sys.stdout.flush()


if __name__ == "__main__":
    main()
