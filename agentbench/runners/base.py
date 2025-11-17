"""Base runner abstractions."""

from __future__ import annotations

from typing import Protocol

from agentbench.context import EvalContext
from agentbench.types import Case, RunnerOutput


class Runner(Protocol):
    """Protocol for runners that execute test cases."""

    async def run_case(self, case: Case, ctx: EvalContext) -> RunnerOutput:
        """
        Execute a test case and return the result.

        Args:
            case: Test case to execute
            ctx: Evaluation context

        Returns:
            RunnerOutput with output, error, and metrics
        """
        ...
