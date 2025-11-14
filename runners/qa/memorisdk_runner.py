#!/usr/bin/env python3
import asyncio
import json
import os
import sys
from contextlib import suppress
from pathlib import Path
from uuid import uuid4
from memori import Memori
from openai import OpenAI

async def main_async():
    case = json.loads(sys.stdin.read())
    model = os.getenv("MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    
    db_root = Path("runs") / "memori"
    db_root.mkdir(parents=True, exist_ok=True)
    namespace = f"agentbench-{model}-{uuid4().hex}"
    db_path = db_root / f"{namespace}.db"
    
    # Initialize Memori with auto_ingest (skip conscious_ingest for speed)
    memori = Memori(
        namespace=namespace,
        database_connect=f"sqlite:///{db_path.resolve()}",
        conscious_ingest=False,  # Skip manual recording - too slow
        auto_ingest=True,
        base_url=base_url,
        api_key=api_key,
        model=model,
        user_id=case["id"]
    )
    memori.enable()
    
    try:
        question = case["inputs"]["question"]
        choices = case["inputs"].get("choices")
        
        # Use OpenAI client - Memori will auto-inject memories
        client = OpenAI(base_url=base_url, api_key=api_key)
        
        prompt = f"Question: {question}"
        if choices:
            choices_text = "\n".join(f"{key}. {value}" for key, value in sorted(choices.items()))
            prompt += f"\n\nChoices:\n{choices_text}\n\nRespond with only the choice letter (A, B, C, or D)."
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
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
