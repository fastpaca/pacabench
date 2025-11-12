"""GAIA executor using smolagents for agentic tool-based workflows."""

from __future__ import annotations

import os
from typing import Any

from pydantic_evals import increment_eval_metric
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    FinalAnswerTool,
    PythonInterpreterTool,
    VisitWebpageTool,
    WikipediaSearchTool,
)
from smolagents.models import OpenAIServerModel

from memharness.executors.base import Executor

_GAIA_SYSTEM_PROMPT = """You are an expert assistant who can solve any task using code and tools.

You will be given a task to solve. To do so, you have been given access to tools:
these tools are Python functions which you can call with code.

To solve the task, you must plan forward to proceed in a series of steps,
in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences.

At each step:
- In the 'Thought:' sequence: explain your reasoning towards solving the task
- In the 'Code:' sequence: write Python code (must end with '<end_code>')
- The execution results appear in 'Observation:' fields

CRITICAL ANSWER FORMATTING RULES (GAIA Benchmark):
- Numbers: Digits only (42), no commas (1000), no units ($)
- Strings: Minimal words, no articles ('Paris'). Strip whitespace.
- Lists: Comma-separated without spaces ('item1,item2,item3')

When you have completed the task and have a final answer:
1. Ensure your answer follows the formatting rules above
2. Call final_answer(your_answer) to submit your response

CONSTRAINTS:
- Use only defined variables
- Don't chain unpredictable tool calls in single code blocks
- Import only from authorized modules
"""


class GAIAExecutor(Executor):
    """Executor for GAIA benchmark using smolagents with tools.

    This executor uses CodeAgent to solve complex multi-step tasks requiring:
    - Web search and information retrieval
    - Code execution for calculations/data processing
    - Multi-step reasoning with planning

    This is appropriate for testing memory systems on real agentic workflows,
    not just conversational recall.
    """

    def __init__(
        self,
        model: str,
        provider: str,
        max_steps: int = 15,
        planning_interval: int | None = None,
    ):
        """Initialize GAIA executor with agentic capabilities.

        Args:
            model: Model name (e.g., "gpt-4o", "claude-sonnet-4-5")
            provider: Provider name (e.g., "openai", "anthropic")
            max_steps: Maximum number of agent steps (default: 15)
            planning_interval: Steps between replanning (None = no replanning)
        """
        self.model = model
        self.provider = provider
        self.max_steps = max_steps
        self.planning_interval = planning_interval
        self._agent = None  # Lazy initialization

    def _get_agent(self) -> CodeAgent:
        """Lazy initialize agent with tools (reused across all tasks)."""
        if self._agent is not None:
            return self._agent

        # Initialize model based on provider
        if self.provider == "openai":
            model_obj = OpenAIServerModel(
                model_id=self.model,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        elif self.provider == "anthropic":
            # smolagents uses OpenAI-compatible API format for Anthropic
            model_obj = OpenAIServerModel(
                model_id=self.model,
                api_base="https://api.anthropic.com/v1",
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        # Initialize tools for GAIA
        tools = [
            DuckDuckGoSearchTool(),  # Web search (no API key needed)
            WikipediaSearchTool(),  # Structured knowledge
            VisitWebpageTool(),  # Detailed webpage content
            PythonInterpreterTool(),  # Code execution
            FinalAnswerTool(),  # Submit final answer
        ]

        # Create CodeAgent
        # Note: system_prompt customization requires passing it via model initialization
        # For now, using default CodeAgent prompt which is already optimized
        self._agent = CodeAgent(
            tools=tools,
            model=model_obj,
            max_steps=self.max_steps,
            planning_interval=self.planning_interval,
            verbosity_level=0,  # Quiet mode for benchmarking
        )

        return self._agent

    async def execute(self, inputs: dict[str, Any]) -> str:
        """Execute GAIA task with agentic tool-based approach.

        Args:
            inputs: Task inputs with 'question' key

        Returns:
            String output (final answer)
        """
        agent = self._get_agent()

        # Get question
        question = inputs.get("question", "")

        # Handle file attachments if present
        if inputs.get("file_name"):
            # TODO: Add file processing support
            # For now, note the file exists but agent can't access it
            question += (
                f"\n\nNote: A file '{inputs['file_name']}' is mentioned but not yet accessible."
            )

        # Run agent with full result tracking
        try:
            result = agent.run(question, return_full_result=True)

            # Track token usage
            increment_eval_metric("input_tokens", result.token_usage.input_tokens)
            increment_eval_metric("output_tokens", result.token_usage.output_tokens)

            # Track number of agent steps (turns/iterations)
            increment_eval_metric("agent_steps", len(result.steps))

            # Mark as successful execution
            increment_eval_metric("agent_runs", 1)

            # Extract final answer
            return str(result.output).strip()

        except Exception as e:
            # If agent fails, return error message
            increment_eval_metric("agent_errors", 1)
            return f"Error: {str(e)}"
