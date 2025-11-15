"""Stage 1: Input - Test case definition."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Dataset(Enum):
    """Available datasets."""

    MEMBENCH = "membench"
    LONGMEMEVAL = "longmemeval"
    GAIA = "gaia"


@dataclass
class Case:
    """A single test case."""

    id: str
    task_type: str
    inputs: dict[str, Any]
    expected_output: str
    metadata: dict[str, Any]
