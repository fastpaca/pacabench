#!/usr/bin/env python3
"""
Long-context agentic runner.

Uses smolagents CodeAgent with tools for multi-step reasoning.
Reads test case from stdin, outputs result to stdout.
"""

import json
import os
import sys

from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    FinalAnswerTool,
    PythonInterpreterTool,
    VisitWebpageTool,
    WikipediaSearchTool,
)
from smolagents.models import OpenAIServerModel


def main() -> None:
    """Main runner entry point."""
    case = json.loads(sys.stdin.read())

    model = os.getenv("MODEL", "gpt-4o")
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")

    model_obj = OpenAIServerModel(
        model_id=model,
        api_base=base_url,
        api_key=api_key,
    )

    tools = [
        DuckDuckGoSearchTool(),
        WikipediaSearchTool(),
        VisitWebpageTool(),
        PythonInterpreterTool(),
        FinalAnswerTool(),
    ]

    agent = CodeAgent(
        tools=tools,
        model=model_obj,
        max_steps=15,
        verbosity_level=0,
    )

    question = case["inputs"]["question"]

    if case["inputs"].get("file_name"):
        question += (
            f"\n\nNote: A file '{case['inputs']['file_name']}' is mentioned but not yet accessible."
        )

    try:
        result = agent.run(question, return_full_result=True)
        answer = str(result.output).strip()

        print(json.dumps({"result": answer, "error": None}))
    except Exception as e:
        print(json.dumps({"result": None, "error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
