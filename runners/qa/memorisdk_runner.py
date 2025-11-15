#!/usr/bin/env python3
import asyncio
import json
import os
import sys
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

from memori import Memori
from openai import OpenAI


def _build_messages(
    conversation: list[dict[str, Any]] | None,
    question: str,
    choices: dict[str, str] | None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []

    if conversation:
        for turn in conversation:
            role = turn.get("role")
            content = turn.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                messages.append({"role": role, "content": content.strip()})

    prompt = f"Question: {question}"
    if choices:
        choices_text = "\n".join(f"{key}. {value}" for key, value in sorted(choices.items()))
        prompt += (
            f"\n\nChoices:\n{choices_text}\n\nRespond with only the choice letter (A, B, C, or D)."
        )

    messages.append({"role": "user", "content": prompt})
    return messages


async def main_async():
    case = json.loads(sys.stdin.read())
    model = os.getenv("MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")

    run_id = os.getenv("AGENTBENCH_RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S")
    dataset = os.getenv("AGENTBENCH_DATASET", "agentbench")
    namespace = f"{dataset}-{model}-{run_id}"

    db_root = Path("runs") / "memori" / namespace
    db_root.mkdir(parents=True, exist_ok=True)
    db_path = db_root / "memori.db"

    # Enable Memori so it can automatically handle recall/ingest around the model call
    memori = Memori(
        namespace=namespace,
        database_connect=f"sqlite:///{db_path.resolve()}",
        conscious_ingest=True,
        auto_ingest=True,
        base_url=base_url,
        api_key=api_key,
        model=model,
        user_id=case["id"],
    )
    memori.enable()

    try:
        question = case["inputs"]["question"]
        choices = case["inputs"].get("choices")
        conversation = case["inputs"].get("conversation")

        messages = _build_messages(conversation, question, choices)

        client = OpenAI(base_url=base_url, api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )

        result = response.choices[0].message.content or ""
        print(json.dumps({"result": result, "error": None}))
    except Exception as e:
        print(json.dumps({"result": None, "error": str(e)}))
        sys.exit(1)
    finally:
        with suppress(Exception):
            memori.disable()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
