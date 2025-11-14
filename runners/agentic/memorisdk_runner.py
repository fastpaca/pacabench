#!/usr/bin/env python3
import json
import os
import sys
from contextlib import suppress
from pathlib import Path
from uuid import uuid4
from memori import Memori
from smolagents import CodeAgent, DuckDuckGoSearchTool, FinalAnswerTool, PythonInterpreterTool, VisitWebpageTool, WikipediaSearchTool
from smolagents.models import OpenAIServerModel

def main():
    case = json.loads(sys.stdin.read())
    model = os.getenv("MODEL", "gpt-4o")
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    
    db_root = Path("runs") / "memori"
    db_root.mkdir(parents=True, exist_ok=True)
    namespace = f"agentbench-gaia-{model}-{uuid4().hex}"
    db_path = db_root / f"{namespace}.db"
    
    memori = Memori(namespace=namespace, database_connect=f"sqlite:///{db_path.resolve()}", conscious_ingest=True, auto_ingest=True, base_url=base_url, model=model, user_id=case["id"])
    memori.enable()
    
    try:
        model_obj = OpenAIServerModel(model_id=model, api_base=base_url, api_key=api_key)
        tools = [DuckDuckGoSearchTool(), WikipediaSearchTool(), VisitWebpageTool(), PythonInterpreterTool(), FinalAnswerTool()]
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
