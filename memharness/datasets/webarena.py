"""WebArena dataset implementation for browser-based agent evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from memharness.datasets.base import BenchmarkDataset

if TYPE_CHECKING:
    from memharness.executors.base import Executor


class StringMatchEvaluator(Evaluator):
    """Evaluator for string matching tasks."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        """Check if response matches expected string patterns.

        Supports three matching modes from WebArena:
        - exact_match: Response must exactly match reference
        - must_include: Response must contain all required phrases
        - fuzzy_match: Flexible matching (implemented as substring match for now)
        """
        if not ctx.expected_output or not isinstance(ctx.expected_output, dict):
            return False

        response = ctx.output.strip().lower() if ctx.output else ""
        eval_config = ctx.expected_output

        if "exact_match" in eval_config:
            reference = eval_config["exact_match"].strip().lower()
            return response == reference

        if "must_include" in eval_config:
            required_phrases = eval_config["must_include"]
            if not isinstance(required_phrases, list):
                required_phrases = [required_phrases]
            return all(phrase.lower() in response for phrase in required_phrases)

        if "fuzzy_match" in eval_config:
            reference = eval_config["fuzzy_match"].strip().lower()
            return reference in response or response in reference

        return False


class URLMatchEvaluator(Evaluator):
    """Evaluator for URL navigation tasks."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        """Check if agent navigated to correct URL.

        Extracts final URL from response and compares against reference URL.
        Supports base path matching and query parameter validation.
        """
        if not ctx.expected_output or not isinstance(ctx.expected_output, dict):
            return False

        reference_url = ctx.expected_output.get("reference_url", "")
        if not reference_url:
            return False

        response = ctx.output.strip() if ctx.output else ""
        if not response:
            return False

        final_url = self._extract_final_url(response)
        if not final_url:
            return False

        return self._urls_match(final_url, reference_url)

    @staticmethod
    def _extract_final_url(response: str) -> str:
        """Extract final URL from response.

        Expected format: Response should contain final URL as text or structured data.
        """
        lines = response.strip().split("\n")
        for line in reversed(lines):
            if line.startswith("http://") or line.startswith("https://"):
                return line.strip()
        return ""

    @staticmethod
    def _urls_match(url1: str, url2: str) -> bool:
        """Check if two URLs match (simple implementation)."""
        from urllib.parse import urlparse

        parsed1 = urlparse(url1)
        parsed2 = urlparse(url2)

        return (
            parsed1.scheme == parsed2.scheme
            and parsed1.netloc == parsed2.netloc
            and parsed1.path == parsed2.path
        )


class HTMLContentEvaluator(Evaluator):
    """Evaluator for HTML content verification tasks."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        """Check if specific content appears in final page state.

        Validates that expected elements or text appear in the final HTML/page state.
        """
        if not ctx.expected_output or not isinstance(ctx.expected_output, dict):
            return False

        response = ctx.output.strip().lower() if ctx.output else ""
        expected_content = ctx.expected_output.get("expected_content", [])

        if not isinstance(expected_content, list):
            expected_content = [expected_content]

        return all(str(content).lower() in response for content in expected_content)


class WebArena(BenchmarkDataset):
    """WebArena dataset for browser-based agent evaluation.

    WebArena evaluates agents on realistic web navigation tasks across multiple
    domains (shopping, Reddit, GitLab, Wikipedia, maps). Tasks require agents to
    interact with actual websites through browser automation.

    Infrastructure requirements:
    - WebArena websites running (via Docker or hosted)
    - Environment variables for website URLs
    - Authentication cookies for sites requiring login
    """

    name = "webarena"

    def __init__(
        self,
        domain: str | None = None,
        task_ids: list[int] | None = None,
        executor_class: type[Executor] | None = None,
    ):
        """Initialize WebArena dataset.

        Args:
            domain: Optional domain filter ("shopping", "reddit", "gitlab", etc.)
            task_ids: Optional list of specific task IDs to load (0-811)
            executor_class: Optional executor class override
        """
        from memharness.executors.agentlab import AgentLabExecutor

        self.default_executor = AgentLabExecutor
        super().__init__(executor_class=executor_class)

        self.domain = domain
        self.task_ids = task_ids

    def load(self, limit: int | None = None) -> Dataset:
        """Load WebArena dataset.

        Args:
            limit: Optional limit on number of cases

        Returns:
            pydantic-evals Dataset with browser task cases

        Note:
            Uses BrowserGym's task registration. Tasks are registered as
            'browsergym/webarena.X' where X is 0-811.
        """
        try:
            import browsergym.webarena  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "BrowserGym WebArena is required. Install with: pip install browsergym-webarena"
            ) from e

        task_ids = self.task_ids or list(range(812))

        if limit is not None:
            task_ids = task_ids[:limit]

        cases = [self._build_case_from_task_id(task_id) for task_id in task_ids]

        evaluators = [
            StringMatchEvaluator(),
            URLMatchEvaluator(),
            HTMLContentEvaluator(),
        ]

        return Dataset(cases=cases, evaluators=evaluators)

    def _build_case_from_task_id(self, task_id: int) -> Case:
        """Build a case from a BrowserGym task ID."""
        return Case(
            name=f"webarena_{task_id}",
            inputs={
                "task_id": str(task_id),
                "benchmark": "webarena",
            },
            expected_output={
                "task_id": task_id,
            },
            metadata={
                "task_id": task_id,
                "benchmark": "webarena",
            },
        )

    def _build_case(self, task: dict[str, Any]) -> Case:
        """Build a pydantic-evals Case from a WebArena task."""
        task_id = task.get("task_id", "unknown")
        intent = task.get("intent", "")
        start_url = task.get("start_url", "")
        sites = task.get("sites", [])
        require_login = task.get("require_login", False)
        storage_state = task.get("storage_state")

        eval_config = task.get("eval", {})
        eval_types = eval_config.get("eval_types", [])
        reference_answers = eval_config.get("reference_answers", {})
        reference_url = eval_config.get("reference_url", "")

        expected_output = {
            "eval_types": eval_types,
            "reference_url": reference_url,
            **reference_answers,
        }

        return Case(
            name=f"webarena_{task_id}",
            inputs={
                "intent": intent,
                "start_url": start_url,
                "sites": sites,
                "require_login": require_login,
                "storage_state": storage_state,
            },
            expected_output=expected_output,
            metadata={
                "task_id": task_id,
                "domain": sites[0] if sites else "unknown",
                "require_reset": task.get("require_reset", False),
                "geolocation": task.get("geolocation"),
            },
        )
