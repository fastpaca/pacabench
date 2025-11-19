import json
import os
import sys

from mem0 import Memory
from openai import OpenAI


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
                "collection_name": "agentbench",
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
        # We fail fast here because we want to ensure we are testing the correct setup
        sys.exit(1)

    # Initialize OpenAI client for response generation
    client = OpenAI(api_key=api_key, base_url=proxy_url)

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        question = data.get("input", "")
        case_id = data.get("case_id")

        # Use case_id as user_id to isolate memories per case
        user_id = str(case_id)

        try:
            # 1. Populate memory with history
            history = data.get("history", [])
            messages_to_add = []
            for msg in history:
                if isinstance(msg, dict):
                    role = msg.get("role")
                    content = msg.get("content")
                    # Handle alternative formats if any (agentbench usually sends role/content)
                    if not role or not content:
                        u = msg.get("user_message") or msg.get("user")
                        a = msg.get("assistant_message") or msg.get("assistant")
                        if u:
                            messages_to_add.append({"role": "user", "content": u})
                        if a:
                            messages_to_add.append({"role": "assistant", "content": a})
                    else:
                        messages_to_add.append({"role": role, "content": content})

            if messages_to_add:
                m.add(messages_to_add, user_id=user_id)

            # 2. Search memory
            search_result = m.search(query=question, user_id=user_id, limit=5)

            # Robustly parse search results (mirroring qa_mem0 logic)
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

            # 3. Generate Response
            prompt_content = f"{memory_context}Question: {question}"

            # Handle choices if present
            choices = data.get("choices")
            if choices and isinstance(choices, dict):
                choices_text = "\n".join(
                    f"{key}. {value}" for key, value in sorted(choices.items())
                )
                prompt_content += f"\n\nChoices:\n{choices_text}\n\nRespond with only the choice letter (A, B, C, or D)."

            response = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt_content}]
            )
            output = response.choices[0].message.content

            print(json.dumps({"output": output, "error": None}))
        except Exception as e:
            sys.stderr.write(f"Error in agent loop: {e}\n")
            print(json.dumps({"output": None, "error": str(e)}))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
