#!/usr/bin/env python3
import json
import os
import sys
from contextlib import suppress
from datetime import datetime
from pathlib import Path

from memori import Memori
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    FinalAnswerTool,
    PythonInterpreterTool,
    VisitWebpageTool,
    WikipediaSearchTool,
)
from smolagents.models import OpenAIServerModel


def main():
    case = json.loads(sys.stdin.read())
    model = os.getenv("MODEL", "gpt-4o")
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")

    run_id = os.getenv("AGENTBENCH_RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S")
    dataset = os.getenv("AGENTBENCH_DATASET", "agentbench")
    namespace = f"{dataset}-{model}-{run_id}"

    db_root = Path("runs") / "memori" / namespace
    db_root.mkdir(parents=True, exist_ok=True)
    db_path = db_root / "memori.db"

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
        model_obj = OpenAIServerModel(model_id=model, api_base=base_url, api_key=api_key)
        tools = [
            DuckDuckGoSearchTool(),
            WikipediaSearchTool(),
            VisitWebpageTool(),
            PythonInterpreterTool(),
            FinalAnswerTool(),
        ]
        agent = CodeAgent(tools=tools, model=model_obj, max_steps=15, verbosity_level=0)
        question = case["inputs"]["question"]
        if case["inputs"].get("file_name"):
            question += f"\n\nNote: A file '{case['inputs']['file_name']}' is mentioned but not yet accessible."
        result = agent.run(question, return_full_result=True)
        answer = str(result.output).strip()
        memori.record_conversation(question, answer, model=model)
        print(json.dumps({"result": answer, "error": None}))
    except Exception as e:
        print(json.dumps({"result": None, "error": str(e)}))
        sys.exit(1)
    finally:
        with suppress(Exception):
            memori.disable()


if __name__ == "__main__":
    main()
