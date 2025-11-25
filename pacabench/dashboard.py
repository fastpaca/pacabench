import time

from pydantic import BaseModel, Field, PrivateAttr
from rich.console import Group
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text


class AgentDatasetState(BaseModel):
    agent_name: str
    dataset_name: str
    status: str = "Pending"  # Pending, Running, Completed
    total_cases: int = 0
    completed_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    error_cases: int = 0
    total_cost: float = 0.0
    last_case_id: str | None = None
    avg_latency_ms: float = 0.0

    # Private attributes for running calculations
    _latency_sum: float = PrivateAttr(default=0.0)
    _latency_count: int = PrivateAttr(default=0)

    def update_metrics(
        self, passed: bool, error: bool, cost: float, latency_ms: float, case_id: str
    ):
        self.completed_cases += 1
        self.total_cost += cost
        self.last_case_id = case_id
        if error:
            self.error_cases += 1
        elif passed:
            self.passed_cases += 1
        else:
            self.failed_cases += 1

        if latency_ms > 0:
            self._latency_sum += latency_ms
            self._latency_count += 1
            self.avg_latency_ms = self._latency_sum / self._latency_count

    @property
    def pass_rate(self) -> float:
        if self.completed_cases == 0:
            return 0.0
        # Errors are usually excluded from pass rate denominator in some views,
        # but let's use completed_cases (which includes errors) for robustness
        # or just (passed + failed). Let's use passed / completed for simple "Success Rate"
        return (self.passed_cases / self.completed_cases) * 100


class DashboardState(BaseModel):
    start_time: float = Field(default_factory=time.time)
    total_cost: float = 0.0
    circuit_open: bool = False
    # Key: f"{agent_name}/{dataset_name}"
    agent_states: dict[str, AgentDatasetState] = Field(default_factory=dict)

    def get_state(self, agent: str, dataset: str) -> AgentDatasetState:
        key = f"{agent}/{dataset}"
        if key not in self.agent_states:
            self.agent_states[key] = AgentDatasetState(agent_name=agent, dataset_name=dataset)
        return self.agent_states[key]

    def init_agent(self, agent: str, dataset: str, total_cases: int):
        s = self.get_state(agent, dataset)
        s.total_cases = total_cases
        s.status = "Pending"


class DashboardRenderer:
    def __init__(self):
        self.overall_progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("{task.completed}/{task.total}"),
            expand=True,
        )
        self.overall_task_id = self.overall_progress.add_task("Overall Progress", total=0)

    def render(self, state: DashboardState) -> Group:
        # Update Overall Progress
        total_cases = 0
        completed_cases = 0
        for s in state.agent_states.values():
            total_cases += s.total_cases
            completed_cases += s.completed_cases

        self.overall_progress.update(
            self.overall_task_id, total=total_cases, completed=completed_cases
        )

        # Header Stats
        elapsed = time.time() - state.start_time
        header_text = (
            f"[bold]Elapsed:[/bold] {elapsed:.1f}s  "
            f"[bold]Total Cost:[/bold] ${state.total_cost:.4f}  "
            f"[bold]Circuit Breaker:[/bold] {'[red]TRIPPED[/red]' if state.circuit_open else '[green]OK[/green]'}"
        )

        # Agent Table
        table = Table(expand=True, box=None, padding=(0, 1))
        table.add_column("Agent / Dataset", style="cyan", ratio=3)
        table.add_column("Status", ratio=2)
        table.add_column("Progress", justify="right", ratio=2)
        table.add_column("Pass Rate", justify="right", ratio=2)
        table.add_column("Cost", justify="right", ratio=2)
        table.add_column("Latency (Avg)", justify="right", ratio=2)
        table.add_column("Last Case", style="dim", ratio=2)

        # Sort by agent name
        sorted_keys = sorted(state.agent_states.keys())

        for key in sorted_keys:
            s = state.agent_states[key]

            # Progress string
            prog_str = f"{s.completed_cases}/{s.total_cases}"

            # Status style
            status_style = "dim"
            if s.status == "Running":
                status_style = "bold green"
            elif s.status == "Completed":
                status_style = "bold blue"

            # Pass rate color
            pass_rate = s.pass_rate
            pass_color = "green"
            if pass_rate < 50:
                pass_color = "red"
            elif pass_rate < 80:
                pass_color = "yellow"

            table.add_row(
                f"{s.agent_name}\n[dim]{s.dataset_name}[/dim]",
                f"[{status_style}]{s.status}[/{status_style}]",
                prog_str,
                f"[{pass_color}]{pass_rate:.1f}%[/{pass_color}]",
                f"${s.total_cost:.4f}",
                f"{s.avg_latency_ms:.0f}ms",
                s.last_case_id or "-",
            )

        return Group(
            Text.from_markup(header_text),
            Text(),
            self.overall_progress,
            Text(),
            table,
        )
