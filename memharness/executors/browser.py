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
        """
        agent = self._get_agent()
        await self._init_browser()

        try:
            intent = inputs.get("intent", "")
            start_url = inputs.get("start_url", "")
            storage_state = inputs.get("storage_state")

            page = await self._browser.new_page(storage_state=storage_state)

            if start_url:
                await page.goto(start_url, wait_until="domcontentloaded")

            max_steps = 15
            step_count = 0
            action_history = []

            while step_count < max_steps:
                step_count += 1

                page_state = await self._get_page_state(page)

                history_text = "\n".join(
                    f"Step {i + 1}: {action}" for i, action in enumerate(action_history)
                )

                prompt = f"""You are a web automation agent. Your task is:

{intent}

Current page: {page.url}
Page title: {page_state["title"]}

Previous actions:
{history_text if history_text else "None"}

Current page content (simplified):
{page_state["content"][:2000]}

Decide what to do next. You can:
- CLICK <element_text> - Click on an element by its visible text
- TYPE <element_text> <text> - Type text into an input field
- NAVIGATE <url> - Navigate to a URL
- SCROLL - Scroll down the page
- DONE - Task is complete

Respond with ONLY one action command, or DONE if the task is complete."""

                result = await agent.run(prompt, message_history=self._build_message_history())

                usage = result.usage()
                increment_eval_metric("input_tokens", usage.input_tokens)
                increment_eval_metric("output_tokens", usage.output_tokens)
                increment_eval_metric("cache_write_tokens", usage.cache_write_tokens)
                increment_eval_metric("cache_read_tokens", usage.cache_read_tokens)

                action = str(result.output).strip()
                action_history.append(action)

                increment_eval_metric("browser_actions", 1)

                if action.upper().startswith("DONE"):
                    break

                try:
                    await self._execute_action(page, action)
                except Exception as e:
                    action_history.append(f"[ERROR] {str(e)}")

                await page.wait_for_timeout(500)

            page_title = await page.title()
            final_url = page.url
            final_content = await page.content()

            await page.close()

            increment_eval_metric("total_steps", step_count)

            return f"{final_url}\n{page_title}\n{final_content[:500]}"

        finally:
            await self._cleanup_browser()

    async def _get_page_state(self, page) -> dict:
        """Extract current page state."""
        title = await page.title()
        content = await page.inner_text("body")
        url = page.url

        return {"title": title, "content": content, "url": url}

    async def _execute_action(self, page, action: str):
        """Execute a browser action."""
        action_upper = action.upper()

        if action_upper.startswith("CLICK "):
            element_text = action[6:].strip()
            await page.get_by_text(element_text, exact=False).first.click()

        elif action_upper.startswith("TYPE "):
            parts = action[5:].split(maxsplit=1)
            if len(parts) == 2:
                element_text, text_to_type = parts
                await page.get_by_text(element_text, exact=False).first.fill(text_to_type)

        elif action_upper.startswith("NAVIGATE "):
            url = action[9:].strip()
            await page.goto(url, wait_until="domcontentloaded")

        elif action_upper.startswith("SCROLL"):
            await page.evaluate("window.scrollBy(0, window.innerHeight)")

    def _build_message_history(self) -> list:
        """Build message history for agent context (override in subclasses for memory)."""
        return []
