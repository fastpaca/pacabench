from typing import Protocol

from rich.live import Live

from pacabench.dashboard import DashboardRenderer, DashboardState


class ProgressReporter(Protocol):
    """Protocol for reporting benchmark progress."""

    async def start(self, state: DashboardState) -> None:
        """Start the reporter (e.g. initialize live display)."""
        ...

    async def update(self, state: DashboardState) -> None:
        """Update the display with new state."""
        ...

    async def stop(self) -> None:
        """Stop the reporter (e.g. clean up display)."""
        ...


class RichProgressReporter:
    """Reporter that uses Rich Live display for CLI output."""

    def __init__(self):
        self.renderer = DashboardRenderer()
        self.live: Live | None = None

    async def start(self, state: DashboardState) -> None:
        self.live = Live(self.renderer.render(state), refresh_per_second=4)
        self.live.start()

    async def update(self, state: DashboardState) -> None:
        if self.live:
            self.live.update(self.renderer.render(state))

    async def stop(self) -> None:
        if self.live:
            self.live.stop()
