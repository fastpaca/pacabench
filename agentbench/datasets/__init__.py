"""Dataset loaders for AgentBench."""

from agentbench.datasets.gaia import load_gaia
from agentbench.datasets.longmemeval import load_longmemeval
from agentbench.datasets.membench import load_membench
from agentbench.stages.case import Case, Dataset


def load_dataset(
    dataset: Dataset | str,
    limit: int | None = None,
) -> list[Case]:
    """
    Load dataset by name.

    Args:
        dataset: Dataset to load (Dataset enum or string name)
        limit: Optional limit on number of samples

    Returns:
        List of Cases
    """
    if isinstance(dataset, str):
        try:
            dataset = Dataset(dataset)
        except ValueError as e:
            raise ValueError(f"Unknown dataset: {dataset}") from e

    if dataset == Dataset.MEMBENCH:
        return load_membench(limit=limit)
    elif dataset == Dataset.LONGMEMEVAL:
        return load_longmemeval(limit=limit)
    elif dataset == Dataset.GAIA:
        return load_gaia(limit=limit)
    else:
        raise ValueError(f"Unhandled dataset: {dataset}")
