from collections.abc import Callable
from typing import Any, Protocol

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


class TextualProgressReporter:
    """Reporter that pushes updates to a Textual App via callback."""

    def __init__(self, update_callback: Callable[[DashboardState], Any]):
        self.update_callback = update_callback

    async def start(self, state: DashboardState) -> None:
        if asyncio.iscoroutinefunction(self.update_callback):
            await self.update_callback(state)
        else:
            self.update_callback(state)

    async def update(self, state: DashboardState) -> None:
        if asyncio.iscoroutinefunction(self.update_callback):
            await self.update_callback(state)
        else:
            self.update_callback(state)

    async def stop(self) -> None:
        pass
