"""Long-context browser agent using AgentLab/BrowserGym."""

from __future__ import annotations

from httpx import AsyncClient, Limits
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

try:
    from browsergym.core.action.highlevel import HighLevelActionSet
    from browsergym.experiments import Agent

    BROWSERGYM_AVAILABLE = True
except ImportError:
    BROWSERGYM_AVAILABLE = False
    Agent = object  # type: ignore
    HighLevelActionSet = object  # type: ignore

from memharness.executors.agentlab import AgentLabExecutor

_MAX_CONNECTIONS = 20
_MAX_KEEPALIVE_CONNECTIONS = 50
_REQUEST_TIMEOUT_SECONDS = 300.0


class LongContextBrowserAgent(Agent):
    """Long-context browser agent that uses LLMs for reasoning.

    This agent keeps the full conversation history in context and uses
    the LLM to reason about each action. It's the baseline approach without
    external memory systems.
    """

    def __init__(
        self, model: str, provider: str, base_url: str | None = None, api_key: str | None = None
    ):
        """Initialize long-context browser agent.

        Args:
            model: Model name (e.g., "gpt-4o", "claude-sonnet-4-5")
            provider: Provider name ("openai", "anthropic")
            base_url: Optional base URL for custom endpoints
            api_key: Optional API key
        """
        self.action_set = HighLevelActionSet()
        self.model = model
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
        self._pydantic_agent = None

    def _get_pydantic_agent(self) -> PydanticAgent:
        """Lazily initialize PydanticAI agent."""
        if self._pydantic_agent is not None:
            return self._pydantic_agent

        http_client = AsyncClient(
            limits=Limits(
                max_connections=_MAX_CONNECTIONS,
                max_keepalive_connections=_MAX_KEEPALIVE_CONNECTIONS,
            ),
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )

        if self.provider == "anthropic":
            model_obj = AnthropicModel(self.model)
        elif self.provider == "openai":
            model_obj = OpenAIChatModel(
                self.model,
                provider=OpenAIProvider(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    http_client=http_client,
                ),
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        self._pydantic_agent = PydanticAgent(model=model_obj)
        return self._pydantic_agent

    def get_action(self, obs: dict) -> tuple[str, dict]:
        """Get next action based on observation.

        Args:
            obs: Observation dict with:
                - goal: Task description
                - axtree_txt: Accessibility tree
                - last_action: Previous action taken
                - last_action_error: Error from previous action (if any)

        Returns:
            Tuple of (action_string, info_dict)
        """
        import asyncio

        goal = obs.get("goal", "")
        axtree = obs.get("axtree_txt", "")[:2000]
        last_action = obs.get("last_action", "")
        last_error = obs.get("last_action_error", "")

        prompt = f"""You are a web automation agent. Your goal is:

{goal}

Current page (accessibility tree):
{axtree}

Previous action: {last_action if last_action else "None"}
{f"Previous error: {last_error}" if last_error else ""}

Available actions (use Python syntax):
- click('bid') - Click element by ID
- fill('bid', 'text') - Fill input field
- goto('url') - Navigate to URL
- scroll(dx, dy) - Scroll page
- send_msg_to_user('text') - Send message
- noop() - Do nothing

Think step by step about what to do next, then provide ONE action.

Format your response as:
<think>Your reasoning here</think>
<action>action_code()</action>"""

        agent = self._get_pydantic_agent()

        result = asyncio.run(agent.run(prompt))

        response = str(result.output)

        think = self._extract_tag(response, "think")
        action = self._extract_tag(response, "action")

        if not action or action == "":
            action = "noop()"

        info = {
            "think": think,
            "raw_response": response,
            "stats": {
                "goal_length": len(goal),
                "axtree_length": len(axtree),
                "response_length": len(response),
            },
        }

        return action, info

    @staticmethod
    def _extract_tag(text: str, tag: str) -> str:
        """Extract content from XML-style tags."""
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"

        start_idx = text.find(start_tag)
        if start_idx == -1:
            return ""

        start_idx += len(start_tag)
        end_idx = text.find(end_tag, start_idx)

        if end_idx == -1:
            return text[start_idx:].strip()

        return text[start_idx:end_idx].strip()


class LongContextBrowserExecutor(AgentLabExecutor):
    """Long-context browser executor using AgentLab."""

    def _create_agent(self) -> Agent:
        """Create long-context browser agent."""
        return LongContextBrowserAgent(
            model=self.model,
            provider=self.provider,
            base_url=self.kwargs.get("base_url"),
            api_key=self.kwargs.get("api_key"),
        )
