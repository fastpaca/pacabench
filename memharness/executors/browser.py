"""Browser executor for WebArena-style tasks."""

from __future__ import annotations

from httpx import AsyncClient, Limits
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_evals import increment_eval_metric

from memharness.executors.base import Executor

_MAX_CONNECTIONS = 20
_MAX_KEEPALIVE_CONNECTIONS = 50
_REQUEST_TIMEOUT_SECONDS = 300.0


class BrowserExecutor(Executor):
    """Browser automation executor for WebArena tasks.

    This executor handles browser-based tasks where agents need to interact with
    websites. It uses Playwright for browser automation and integrates with
    PydanticAI agents for reasoning.

    Note: This is a baseline implementation. Users should extend or customize
    this class to implement their specific browser automation strategy.
    """

    def __init__(
        self,
        model: str,
        provider: str = "openai",
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ):
        """Initialize browser executor.

        Args:
            model: Model name (e.g., "gpt-4o", "claude-sonnet-4-5")
            provider: Provider name ("openai", "anthropic")
            base_url: Optional base URL for custom endpoints
            api_key: Optional API key
            **kwargs: Additional configuration
        """
        super().__init__(model, provider, **kwargs)
        self.base_url = base_url
        self.api_key = api_key
        self._agent = None
        self._browser = None
        self._playwright = None

    def _get_agent(self) -> Agent:
        """Lazily initialize and return agent."""
        if self._agent is not None:
            return self._agent

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

        self._agent = Agent(model=model_obj)
        return self._agent

    async def _init_browser(self):
        """Initialize Playwright browser."""
        if self._browser is not None:
            return

        try:
            from playwright.async_api import async_playwright
        except ImportError as e:
            raise ImportError(
                "Playwright is required for browser tasks. "
                "Install with: pip install playwright && playwright install"
            ) from e

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=True)

    async def _cleanup_browser(self):
        """Cleanup browser resources."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    async def execute(self, inputs: dict) -> str:
        """Execute browser-based task.

        Args:
            inputs: Dict with:
                - intent: Task description/goal
                - start_url: Initial URL to navigate to
                - sites: List of relevant domains
                - require_login: Whether authentication is needed
                - storage_state: Optional authentication state

        Returns:
            Final page state or result string

        Note: This is a stub implementation. Users should customize the browser
        automation logic based on their specific needs. Key areas to implement:

        1. Browser observation (accessibility tree, HTML, screenshots)
        2. Agent reasoning loop (observe → reason → act)
        3. Action execution (click, type, navigate, etc.)
        4. Memory integration (if using external memory systems)
        5. Task completion detection
        """
        agent = self._get_agent()
        await self._init_browser()

        try:
            intent = inputs.get("intent", "")
            start_url = inputs.get("start_url", "")
            storage_state = inputs.get("storage_state")

            page = await self._browser.new_page(storage_state=storage_state)

            if start_url:
                await page.goto(start_url)

            prompt = f"""You are a web automation agent. Your task is:

{intent}

You are currently at: {start_url}

TODO: Implement browser automation loop here. This should include:
1. Observing the page (accessibility tree, HTML, etc.)
2. Using the agent to reason about next actions
3. Executing actions on the page
4. Repeating until task is complete

For now, this is a stub that returns the page title."""

            result = await agent.run(prompt)

            usage = result.usage()
            increment_eval_metric("input_tokens", usage.input_tokens)
            increment_eval_metric("output_tokens", usage.output_tokens)
            increment_eval_metric("cache_write_tokens", usage.cache_write_tokens)
            increment_eval_metric("cache_read_tokens", usage.cache_read_tokens)

            page_title = await page.title()
            final_url = page.url

            await page.close()

            return f"Task: {intent}\nFinal URL: {final_url}\nPage Title: {page_title}\nAgent Response: {result.output}"

        finally:
            await self._cleanup_browser()
