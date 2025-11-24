import asyncio
import time
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Log,
    ProgressBar,
    Static,
    TabbedContent,
    TabPane,
)

from pacabench.config import load_config
from pacabench.context import build_eval_context
from pacabench.core import Harness
from pacabench.dashboard import DashboardState
from pacabench.persistence import get_failed_case_ids, get_run_summaries
from pacabench.reporters import TextualProgressReporter


class UpdateDashboard(Message):
    """Message to update the dashboard state."""

    def __init__(self, state: DashboardState) -> None:
        self.state = state
        super().__init__()


class StartRun(Message):
    """Message to start a benchmark run."""

    def __init__(
        self,
        config_path: Path,
        agents: list[str],
        datasets: list[str],
        limit: int | None,
        concurrency: int | None,
        whitelist_ids: set[str] | None = None,
    ) -> None:
        self.config_path = config_path
        self.agents = agents
        self.datasets = datasets
        self.limit = limit
        self.concurrency = concurrency
        self.whitelist_ids = whitelist_ids
        super().__init__()


class PreFillLauncher(Message):
    """Message to pre-fill the launcher with retry data."""

    def __init__(self, config_path: Path, whitelist_ids: set[str]) -> None:
        self.config_path = config_path
        self.whitelist_ids = whitelist_ids
        super().__init__()


class DashboardHeader(Static):
    start_time = reactive(0.0)
    total_cost = reactive(0.0)
    circuit_open = reactive(False)
    pass_rate = reactive(0.0)

    def render(self) -> str:
        elapsed = 0.0 if self.start_time == 0 else time.time() - self.start_time
        status = "[red]TRIPPED[/red]" if self.circuit_open else "[green]OK[/green]"
        return (
            f"Elapsed: {elapsed:.1f}s | "
            f"Cost: ${self.total_cost:.4f} | "
            f"Pass Rate: {self.pass_rate:.1f}% | "
            f"Circuit: {status}"
        )

    def on_mount(self):
        self.set_interval(1, self.refresh)


class LauncherView(Container):
    def __init__(self, config_path: Path = Path("agentbench.yaml"), **kwargs):
        super().__init__(**kwargs)
        self.config_path = config_path
        self.loaded_config = None
        self.whitelist_ids: set[str] | None = None
        if self.config_path.exists():
            try:
                self.loaded_config = load_config(self.config_path)
            except Exception:
                pass

    def compose(self) -> ComposeResult:
        yield Label("Configuration File", classes="section-label")
        yield Input(str(self.config_path), id="config_path")

        yield Label("Agents", classes="section-label")
        with VerticalScroll(id="agents_list", classes="box"):
            if self.loaded_config:
                for agent in self.loaded_config.agents:
                    yield Checkbox(agent.name, value=True, id=f"agent_{agent.name}")
            else:
                yield Label("No config loaded")

        yield Label("Datasets", classes="section-label")
        with VerticalScroll(id="datasets_list", classes="box"):
            if self.loaded_config:
                for ds in self.loaded_config.datasets:
                    yield Checkbox(ds.name, value=True, id=f"dataset_{ds.name}")

        yield Label("Parameters", classes="section-label")
        with Horizontal():
            yield Label("Limit:", classes="param-label")
            yield Input(placeholder="All", id="limit_input", classes="param-input")
            yield Label("Concurrency:", classes="param-label")
            yield Input(
                str(self.loaded_config.config.concurrency) if self.loaded_config else "4",
                id="concurrency_input",
                classes="param-input",
            )

        yield Label("", id="retry_info", classes="retry-label")
        with Horizontal(classes="action-bar"):
            yield Button("Start Run", variant="primary", id="start_btn")
            yield Button("Clear Retry", variant="error", id="clear_retry_btn", classes="hidden")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start_btn":
            self.action_start_run()
        elif event.button.id == "clear_retry_btn":
            self.clear_retry()

    def action_start_run(self) -> None:
        config_path = Path(self.query_one("#config_path", Input).value)

        agents = []
        for cb in self.query(Checkbox):
            if cb.value and cb.id and cb.id.startswith("agent_"):
                agents.append(cb.id.replace("agent_", ""))

        datasets = []
        for cb in self.query(Checkbox):
            if cb.value and cb.id and cb.id.startswith("dataset_"):
                datasets.append(cb.id.replace("dataset_", ""))

        limit_val = self.query_one("#limit_input", Input).value
        limit = int(limit_val) if limit_val.strip() else None

        conc_val = self.query_one("#concurrency_input", Input).value
        concurrency = int(conc_val) if conc_val.strip() else None

        self.post_message(
            StartRun(
                config_path=config_path,
                agents=agents,
                datasets=datasets,
                limit=limit,
                concurrency=concurrency,
                whitelist_ids=self.whitelist_ids,
            )
        )

    def set_retry_mode(self, whitelist_ids: set[str]):
        self.whitelist_ids = whitelist_ids
        lbl = self.query_one("#retry_info", Label)
        lbl.update(f"RETRY MODE: Running {len(whitelist_ids)} failed cases only.")
        lbl.classes = "retry-label active"

        self.query_one("#clear_retry_btn").remove_class("hidden")

    def clear_retry(self):
        self.whitelist_ids = None
        lbl = self.query_one("#retry_info", Label)
        lbl.update("")
        lbl.classes = "retry-label"
        self.query_one("#clear_retry_btn").add_class("hidden")


class MonitorView(Container):
    def compose(self) -> ComposeResult:
        yield DashboardHeader(id="monitor_header")
        yield ProgressBar(total=100, show_eta=False, id="overall_progress")
        yield DataTable(id="monitor_table")
        with VerticalScroll(id="log_container"):
            yield Log(id="run_log")

    def on_mount(self):
        table = self.query_one(DataTable)
        table.add_columns(
            "Agent",
            "Dataset",
            "Status",
            "Progress",
            "Pass %",
            "Cost",
            "Latency",
            "Last Case",
        )
        table.cursor_type = "row"

    def update(self, state: DashboardState):
        header = self.query_one(DashboardHeader)
        header.start_time = state.start_time
        header.total_cost = state.total_cost
        header.circuit_open = state.circuit_open

        # Calculate total pass rate
        total_completed = 0
        total_passed = 0
        total_cases = 0

        table = self.query_one(DataTable)

        for key, s in state.agent_states.items():
            total_completed += s.completed_cases
            total_passed += s.passed_cases
            total_cases += s.total_cases

            # Update Table
            row_key = key

            status_style = "dim"
            if s.status == "Running":
                status_style = "bold green"
            elif s.status == "Completed":
                status_style = "bold blue"

            pass_rate = s.pass_rate
            pass_color = "green"
            if pass_rate < 50:
                pass_color = "red"
            elif pass_rate < 80:
                pass_color = "yellow"

            # Row Data
            cols = [
                s.agent_name,
                s.dataset_name,
                f"[{status_style}]{s.status}[/{status_style}]",
                f"{s.completed_cases}/{s.total_cases}",
                f"[{pass_color}]{pass_rate:.1f}%[/{pass_color}]",
                f"${s.total_cost:.4f}",
                f"{s.avg_latency_ms:.0f}ms",
                s.last_case_id or "-",
            ]

            try:
                row_idx = table.get_row_index(row_key)
                for i, val in enumerate(cols):
                    table.update_cell_at(row_idx, i, val)
            except Exception:
                table.add_row(*cols, key=row_key)

        if total_completed > 0:
            header.pass_rate = (total_passed / total_completed) * 100

        bar = self.query_one(ProgressBar)
        if total_cases > 0:
            bar.update(total=total_cases, progress=total_completed)


class HistoryView(Container):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.selected_run_path: Path | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(classes="toolbar"):
            yield Button("Refresh", id="refresh_history")
            yield Button("Retry Failed Cases", id="retry_failed", variant="warning", disabled=True)

        yield DataTable(id="history_table")

    def on_mount(self):
        table = self.query_one(DataTable)
        table.add_columns(
            "Time", "Run ID", "Status", "Progress", "Cases", "Cost", "Datasets", "Agents"
        )
        table.cursor_type = "row"
        self.refresh_history()

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "refresh_history":
            self.refresh_history()
        elif event.button.id == "retry_failed":
            self.action_retry_failed()

    def on_data_table_row_selected(self, event: DataTable.RowSelected):
        # Find the run path from summaries
        # We stored run_id as key
        run_id = event.row_key.value
        # We need to look up path. Ideally we stored it.
        # Let's fetch summaries again or store them in self
        # For simplicity, re-fetch or assume standard path structure?
        # Re-fetching is safer
        runs_dir = Path("runs") # TODO: Configurable?
        if not runs_dir.exists():
            return

        # Quick lookup
        summary = next((s for s in get_run_summaries(runs_dir) if s.run_id == run_id), None)
        if summary:
            self.selected_run_path = summary.path
            self.query_one("#retry_failed").disabled = False
        else:
            self.selected_run_path = None
            self.query_one("#retry_failed").disabled = True

    def refresh_history(self):
        runs_dir = Path("runs")
        if not runs_dir.exists():
            return

        summaries = get_run_summaries(runs_dir)
        table = self.query_one(DataTable)
        table.clear()

        for s in summaries:
            datasets = ",".join(s.datasets) if len(s.datasets) < 3 else f"{len(s.datasets)} datasets"
            agents = ",".join(s.agents) if len(s.agents) < 3 else f"{len(s.agents)} agents"

            table.add_row(
                s.start_time or "-",
                s.run_id,
                s.status,
                f"{s.progress*100:.0f}%" if s.progress is not None else "-",
                f"{s.completed_cases}/{s.total_cases}",
                f"${s.total_cost_usd:.3f}" if s.total_cost_usd is not None else "-",
                datasets,
                agents,
                key=s.run_id
            )

    def action_retry_failed(self):
        if not self.selected_run_path:
            return

        failed_ids = get_failed_case_ids(self.selected_run_path)
        if not failed_ids:
            self.notify("No failed cases found in this run.")
            return

        config_path = self.selected_run_path / "pacabench.yaml"
        if not config_path.exists():
            # Fallback to default if not found
            config_path = Path("agentbench.yaml")

        self.post_message(PreFillLauncher(config_path, failed_ids))


class PacaBenchApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    .box {
        height: auto;
        max-height: 10;
        border: solid $primary;
        margin: 0 1;
    }
    .section-label {
        margin: 1 1 0 1;
        text-style: bold;
    }
    .param-label {
        margin: 1 1;
        content-align: right middle;
    }
    .param-input {
        width: 20;
    }
    #monitor_header {
        height: 3;
        background: $primary;
        color: white;
        content-align: center middle;
    }
    .retry-label {
        color: red;
        text-style: bold;
        margin: 1 2;
    }
    .hidden {
        display: none;
    }
    .action-bar {
        height: auto;
        margin: 1 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="launcher"):
            with TabPane("Launcher", id="launcher"):
                yield LauncherView(id="launcher_view")
            with TabPane("Monitor", id="monitor"):
                yield MonitorView(id="monitor_view")
            with TabPane("History", id="history"):
                yield HistoryView(id="history_view")
        yield Footer()

    def on_start_run(self, message: StartRun) -> None:
        self.query_one(TabbedContent).active = "monitor"
        log = self.query_one("#run_log", Log)
        log.clear()
        log.write(f"Starting run with config: {message.config_path}")

        self.run_benchmark(message)

    def on_pre_fill_launcher(self, message: PreFillLauncher) -> None:
        self.query_one(TabbedContent).active = "launcher"
        launcher = self.query_one(LauncherView)
        launcher.set_retry_mode(message.whitelist_ids)
        # Update config input?
        launcher.query_one("#config_path", Input).value = str(message.config_path)
        self.notify(f"Prepared retry for {len(message.whitelist_ids)} cases.")

    @work(thread=True)
    def run_benchmark(self, message: StartRun):
        try:
            base_cfg = load_config(message.config_path)
            runtime_cfg = base_cfg.model_copy(deep=True)

            # Apply overrides
            if message.concurrency:
                runtime_cfg.config.concurrency = message.concurrency

            if message.agents:
                runtime_cfg.agents = [a for a in runtime_cfg.agents if a.name in message.agents]

            if message.datasets:
                runtime_cfg.datasets = [d for d in runtime_cfg.datasets if d.name in message.datasets]

            ctx = build_eval_context(
                config_path=message.config_path,
                base_config=base_cfg,
                runtime_config=runtime_cfg,
            )

            reporter = TextualProgressReporter(self.update_dashboard)
            harness = Harness(ctx, reporter=reporter)

            asyncio.run(harness.run(limit=message.limit, whitelist_ids=message.whitelist_ids))

            self.call_from_thread(lambda: self.notify("Run Completed!"))

        except Exception as e:
            self.call_from_thread(lambda: self.query_one("#run_log", Log).write(f"Error: {e}"))
            import traceback
            traceback.print_exc()

    def update_dashboard(self, state: DashboardState):
        self.post_message(UpdateDashboard(state))

    def on_update_dashboard(self, message: UpdateDashboard):
        monitor = self.query_one(MonitorView)
        monitor.update(message.state)
