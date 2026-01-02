#!/usr/bin/env python3
"""
Prepare GAIA dataset for PacaBench.

Downloads the GAIA benchmark from HuggingFace, including file attachments,
and converts to JSONL format for use with PacaBench.

Usage:
    uv run python prepare_dataset.py              # All validation cases
    uv run python prepare_dataset.py --level 1    # Level 1 only (easiest)
    uv run python prepare_dataset.py --level 2    # Level 2 only
    uv run python prepare_dataset.py --level 3    # Level 3 only (hardest)

Requires HF_TOKEN environment variable for gated dataset access.
"""
import json
import shutil
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download


def prepare_gaia(
    output_dir: Path,
    split: str = "validation",
    level: int | None = None,
) -> Path:
    """Download and prepare GAIA dataset.

    Args:
        output_dir: Directory to write output files
        split: Dataset split ("validation" or "test")
        level: Optional difficulty level filter (1, 2, or 3)

    Returns:
        Path to the generated JSONL file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    files_dir = output_dir / "files"
    files_dir.mkdir(exist_ok=True)

    print(f"Loading GAIA {split} split...")
    ds = load_dataset("gaia-benchmark/GAIA", "2023_all", split=split)

    if level is not None:
        ds = ds.filter(lambda x: x["Level"] == str(level))
        print(f"Filtered to level {level}: {len(ds)} cases")
    else:
        print(f"Loaded {len(ds)} cases")

    jsonl_path = output_dir / f"{split}.jsonl"
    file_count = 0

    with open(jsonl_path, "w") as f:
        for row in ds:
            case = {
                "case_id": row["task_id"],
                "input": row["Question"],
                "expected": row["Final answer"],
                "level": row["Level"],
            }

            # Download file attachment if present
            if row.get("file_name") and row.get("file_path"):
                try:
                    local_path = hf_hub_download(
                        repo_id="gaia-benchmark/GAIA",
                        filename=row["file_path"],
                        repo_type="dataset",
                    )
                    dest = files_dir / row["file_name"]
                    if not dest.exists():
                        shutil.copy(local_path, dest)
                    case["file_path"] = str(dest.absolute())
                    case["file_name"] = row["file_name"]
                    file_count += 1
                except Exception as e:
                    print(f"  Warning: couldn't download {row['file_name']}: {e}")

            f.write(json.dumps(case) + "\n")

    print(f"Wrote {len(ds)} cases to {jsonl_path}")
    if file_count > 0:
        print(f"Downloaded {file_count} file attachments to {files_dir}")

    return jsonl_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare GAIA dataset for PacaBench")
    parser.add_argument("--output", "-o", default="data", help="Output directory")
    parser.add_argument("--split", default="validation", help="Dataset split")
    parser.add_argument(
        "--level",
        type=int,
        choices=[1, 2, 3],
        help="Filter to specific level (1=easy, 2=medium, 3=hard)",
    )
    args = parser.parse_args()

    prepare_gaia(Path(args.output), args.split, args.level)
