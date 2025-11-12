"""AgentLab/BrowserGym executor for browser-based tasks."""

from __future__ import annotations

import gymnasium as gym
from pydantic_evals import increment_eval_metric

from memharness.executors.base import Executor

try:
    import browsergym.core  # noqa: F401
    import browsergym.webarena  # noqa: F401
    from browsergym.core.action.highlevel import HighLevelActionSet
    from browsergym.experiments import Agent

    AGENTLAB_AVAILABLE = True
except ImportError:
    AGENTLAB_AVAILABLE = False
    Agent = object  # type: ignore
    HighLevelActionSet = object  # type: ignore


class AgentLabExecutor(Executor):
    """Executor for browser tasks using AgentLab/BrowserGym.

    This executor integrates with AgentLab's agent interface and BrowserGym's
    environment system. It allows running agents on WebArena and other browser
    benchmarks.

    Users should subclass this and implement _create_agent() to define their
    agent's behavior (long-context, mem0, zep, etc.).
    """

    def __init__(
        self,
        model: str,
        provider: str,
        max_steps: int = 15,
        headless: bool = True,
        **kwargs,
    ):
        """Initialize AgentLab executor.

        Args:
            model: Model name (e.g., "gpt-4o", "claude-sonnet-4-5")
            provider: Provider name ("openai", "anthropic")
            max_steps: Maximum number of steps per task
            headless: Whether to run browser in headless mode
            **kwargs: Additional configuration
        """
        if not AGENTLAB_AVAILABLE:
            raise ImportError(
                "AgentLab is required for browser tasks. "
                "Install with: pip install agentlab browsergym-core browsergym-webarena"
            )

        super().__init__(model, provider, **kwargs)
        self.max_steps = max_steps
        self.headless = headless
        self._agent = None

    def _create_agent(self) -> Agent:
        """Create AgentLab agent (override in subclasses).

        Subclasses should implement this to create their specific agent type.
        This allows different memory systems to have their own agent implementations.

        Returns:
            BrowserGym Agent instance
        """
        raise NotImplementedError(
            "Subclasses must implement _create_agent() to define agent behavior"
        )

    def _get_agent(self) -> Agent:
        """Lazily initialize and return agent."""
        if self._agent is None:
            self._agent = self._create_agent()
        return self._agent

    async def execute(self, inputs: dict) -> str:
        """Execute browser-based task using BrowserGym.

        Args:
            inputs: Dict with:
                - task_id: WebArena task ID (e.g., "310")
                - benchmark: Benchmark name (e.g., "webarena")

        Returns:
            Final page state or result string
        """
        task_id = inputs.get("task_id", "0")
        benchmark = inputs.get("benchmark", "webarena")

        env_id = f"browsergym/{benchmark}.{task_id}"

        env = gym.make(env_id, headless=self.headless)

        try:
            agent = self._get_agent()

            obs, info = env.reset()

            step_count = 0
            done = False

            while not done and step_count < self.max_steps:
                step_count += 1

                action, agent_info = agent.get_action(obs)

                increment_eval_metric("browser_steps", 1)

                if "think" in agent_info:
                    increment_eval_metric("reasoning_length", len(str(agent_info["think"])))

                obs, reward, terminated, truncated, info = env.step(action)

                done = terminated or truncated

            increment_eval_metric("total_steps", step_count)
            increment_eval_metric("final_reward", float(reward))

            final_url = obs.get("url", "")
            page_content = obs.get("axtree_txt", "")[:1000]

            return f"{final_url}\n{page_content}"

        finally:
            env.close()


class SimpleBrowserAgent(Agent):
    """Simple browser agent for testing.

    This is a minimal agent implementation that can be used as a baseline
    or for testing the AgentLab integration.
    """

    def __init__(self, model: str, provider: str):
        """Initialize simple browser agent.

        Args:
            model: Model name
            provider: Provider name
        """
        self.action_set = HighLevelActionSet()
        self.model = model
        self.provider = provider

    def get_action(self, obs: dict) -> tuple[str, dict]:
        """Get next action based on observation.

        Args:
            obs: Observation dict with goal, axtree_txt, etc.

        Returns:
            Tuple of (action_string, info_dict)
        """
        goal = obs.get("goal", "")
        page_content = obs.get("axtree_txt", "")[:500]

        action = "noop()"

        info = {
            "think": f"Goal: {goal}\nObserving: {page_content[:100]}...",
            "stats": {"goal_length": len(goal), "page_length": len(page_content)},
        }

        return action, info
