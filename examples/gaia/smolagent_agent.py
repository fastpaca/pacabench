#!/usr/bin/env python3
"""
GAIA agent using smolagents framework with tools.

This demonstrates a more capable agent for GAIA that can:
- Search the web (DuckDuckGo)
- Visit and read web pages
- Execute Python code for calculations

Protocol:
- Read JSONL from stdin (one case per line)
- Write JSONL to stdout (one response per line)
- Response must have "output" (success) or "error" (failure)
"""
import json
import sys
import os
from pathlib import Path

# Suppress smolagents console output by setting verbosity to 0
os.environ["SMOLAGENTS_VERBOSITY"] = "0"

from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    LiteLLMModel,
    PythonInterpreterTool,
    VisitWebpageTool,
)


def create_agent():
    """Create a smolagents CodeAgent with tools for GAIA."""
    model = LiteLLMModel(model_id="gpt-5-nano")

    tools = [
        DuckDuckGoSearchTool(),
        VisitWebpageTool(),
        PythonInterpreterTool(),
    ]

    agent = CodeAgent(
        tools=tools,
        model=model,
        max_steps=10,
        verbosity_level=0,  # Suppress console output
    )

    return agent


def solve(agent, question: str, file_path: str | None = None) -> str:
    """Solve a GAIA question using the agent."""
    # Build the task prompt
    task = question

    if file_path and Path(file_path).exists():
        task = f"You have access to a file at: {file_path}\n\n{question}"

    # Add instruction to match GAIA's expected format
    task += "\n\nIMPORTANT: Give ONLY the final answer value, nothing else. Match the exact format requested in the question."

    result = agent.run(task)
    return str(result)


def main():
    """Main loop: read JSONL from stdin, write responses to stdout."""
    import io
    import contextlib

    agent = create_agent()

    for line in sys.stdin:
        if not line.strip():
            continue

        case = json.loads(line)

        # Capture stdout during agent execution to keep JSONL clean
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                answer = solve(
                    agent,
                    question=case["input"],
                    file_path=case.get("file_path"),
                )

            # If we got an answer, return it (ignore warnings/errors in stdout)
            response = {"output": answer}

        except Exception as e:
            captured_output = captured.getvalue()
            error_msg = str(e)
            if captured_output:
                error_msg = f"{captured_output.strip()}\n{error_msg}"
            response = {"error": error_msg, "error_type": "system_failure"}

        # Only print our JSONL response
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
