import time

from textual.app import App, ComposeResult
from textual.reactive import reactive
from textual.widgets import DataTable, Footer, Header, ProgressBar, Static

from agentbench.dashboard import DashboardState


class DashboardHeader(Static):
    start_time = reactive(0.0)
    total_cost = reactive(0.0)
    circuit_open = reactive(False)

    def render(self) -> str:
        elapsed = 0.0 if self.start_time == 0 else time.time() - self.start_time

        status = "[red]TRIPPED[/red]" if self.circuit_open else "[green]OK[/green]"
        return f"Elapsed: {elapsed:.1f}s | Total Cost: ${self.total_cost:.4f} | Circuit Breaker: {status}"

    def on_mount(self):
        self.set_interval(1, self.refresh)


class AgentBenchApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    DashboardHeader {
        dock: top;
        height: 3;
        content-align: center middle;
        background: $primary;
        color: white;
    }
    ProgressBar {
        dock: top;
        margin: 1 2;
    }
    DataTable {
        margin: 1 2;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        yield DashboardHeader(id="stats")
        yield ProgressBar(total=100, show_eta=False, id="overall_progress")
        yield DataTable(id="agents_table")
        yield Footer()

    def on_mount(self):
        table = self.query_one(DataTable)
        table.add_columns(
            "Agent / Dataset",
            "Status",
            "Progress",
            "Pass Rate",
            "Cost",
            "Latency (Avg)",
            "Last Case",
        )
        # Set cursor type to none so we don't see a highlighted row
        table.cursor_type = "none"

    def update_dashboard(self, state: DashboardState):
        # Update Header
        header = self.query_one(DashboardHeader)
        header.start_time = state.start_time
        header.total_cost = state.total_cost
        header.circuit_open = state.circuit_open

        # Update Overall Progress
        total_cases = 0
        completed_cases = 0
        for s in state.agent_states.values():
            total_cases += s.total_cases
            completed_cases += s.completed_cases

        bar = self.query_one(ProgressBar)
        # Avoid division by zero or invalid updates if totals are 0
        if total_cases > 0:
            bar.update(total=total_cases, progress=completed_cases)

        # Update Table
        table = self.query_one(DataTable)

        # Sort keys for consistent order
        sorted_keys = sorted(state.agent_states.keys())

        for key in sorted_keys:
            s = state.agent_states[key]
            row_key = key

            # Format values
            prog_str = f"{s.completed_cases}/{s.total_cases}"

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

            c1 = f"{s.agent_name}\n[dim]{s.dataset_name}[/dim]"
            c2 = f"[{status_style}]{s.status}[/{status_style}]"
            c3 = prog_str
            c4 = f"[{pass_color}]{pass_rate:.1f}%[/{pass_color}]"
            c5 = f"${s.total_cost:.4f}"
            c6 = f"{s.avg_latency_ms:.0f}ms"
            c7 = s.last_case_id or "-"

            cols = [c1, c2, c3, c4, c5, c6, c7]

            try:
                row_idx = table.get_row_index(row_key)
                for i, val in enumerate(cols):
                    table.update_cell_at(row_idx, i, val)
            except Exception:
                table.add_row(*cols, key=row_key)
