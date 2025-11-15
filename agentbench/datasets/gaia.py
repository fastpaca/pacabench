"""GAIA dataset loader."""

from typing import Any

from datasets import load_dataset

from agentbench.stages.case import Case


def load_gaia(
    level: str = "all",
    split: str = "validation",
    limit: int | None = None,
) -> list[Case]:
    """
    Load GAIA dataset.

    Args:
        level: Difficulty level ("level1", "level2", "level3", or "all")
        split: Dataset split ("validation" or "test")
        limit: Optional limit on number of samples

    Returns:
        List of Cases
    """
    levels = {
        "level1": "2023_level1",
        "level2": "2023_level2",
        "level3": "2023_level3",
        "all": "2023_all",
    }

    if level not in levels:
        raise ValueError(f"Invalid level '{level}'. Available: {list(levels.keys())}")

    dataset_name = levels[level]
    hf_dataset = load_dataset("gaia-benchmark/GAIA", dataset_name, split=split)

    if limit is not None:
        hf_dataset = hf_dataset.select(range(min(limit, len(hf_dataset))))

    cases = []
    for item in hf_dataset:
        inputs: dict[str, Any] = {
            "question": item["Question"],
        }

        if item.get("file_name"):
            inputs["file_name"] = item["file_name"]
            inputs["file_path"] = item.get("file_path", "")

        if item.get("Annotator Metadata"):
            inputs["metadata"] = item["Annotator Metadata"]

        cases.append(
            Case(
                id=item.get("task_id", f"gaia_{len(cases)}"),
                task_type="agentic",
                inputs=inputs,
                expected_output=item["Final answer"],
                metadata={
                    "level": item.get("Level", ""),
                    "file_name": item.get("file_name", ""),
                },
            )
        )

    return cases
