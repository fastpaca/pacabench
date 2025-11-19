import json
import os
import sys

from mem0 import Memory


def main():
    # Initialize Mem0
    # Using Qdrant as vector store (assumed running on localhost:6333)
    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {"host": "localhost", "port": 6333, "collection_name": "mem0_test"},
        }
    }

    try:
        m = Memory.from_config(config)
    except Exception as e:
        sys.stderr.write(f"Failed to init Memory with Qdrant: {e}\n")
        m = Memory()  # Default

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        question = data.get("input", "")
        case_id = data.get("case_id")

        try:
            # Mem0 logic:
            history = data.get("history", [])
            user_id = f"user_{case_id}"

            # Populate memory with history
            for msg in history:
                if isinstance(msg, dict):
                    u = msg.get("user_message") or msg.get("user")
                    a = msg.get("assistant_message") or msg.get("assistant")
                    if u:
                        m.add(u, user_id=user_id, metadata={"role": "user"})
                    if a:
                        m.add(a, user_id=user_id, metadata={"role": "assistant"})

            # Now ask question
            memories = m.search(question, user_id=user_id)
            
            if memories and isinstance(memories, list):
                if isinstance(memories[0], dict):
                    context_str = "\n".join([mem.get("memory", "") for mem in memories])
                elif isinstance(memories[0], str):
                    context_str = "\n".join(memories)
                else:
                    context_str = str(memories)
            else:
                context_str = ""

            # Call LLM (we can use OpenAI directly for this agent implementation)
            from openai import OpenAI

            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

            prompt = f"""Context from memory:
{context_str}

Question: {question}"""

            response = client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
            )
            output = response.choices[0].message.content

            print(json.dumps({"output": output, "error": None}))
        except Exception as e:
            sys.stderr.write(f"Error in agent loop: {e}\n")
            print(json.dumps({"output": None, "error": str(e)}))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
