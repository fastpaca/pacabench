"""MemBench dataset loader."""

import json
from pathlib import Path

from git import Repo
from loguru import logger

from agentbench.stages.case import Case

_MEMBENCH_GITHUB_REPO_URL = "https://github.com/import-myself/Membench.git"
_MEMBENCH_GITHUB_BRANCH = "main"


def _to_conversation(message_list: list) -> list[dict[str, str]]:
    """Convert message_list to a flat conversation."""
    turns = []
    for block in message_list:
        items = block if isinstance(block, list) else [block]
        for item in items:
            if isinstance(item, dict):
                meta_parts = []
                if item.get("time"):
                    meta_parts.append(f"time: {item['time']}")
                if item.get("place"):
                    meta_parts.append(f"place: {item['place']}")
                meta_suffix = f" ({'; '.join(meta_parts)})" if meta_parts else ""

                user_text = item.get("user_message") or item.get("user")
                assistant_text = item.get("assistant_message") or item.get("assistant")

                if user_text:
                    turns.append({"role": "user", "content": f"{user_text}{meta_suffix}"})
                if assistant_text:
                    turns.append({"role": "assistant", "content": f"{assistant_text}{meta_suffix}"})
            elif item:
                turns.append({"role": "user", "content": str(item)})
    return [turn for turn in turns if turn.get("content")]


def _normalize_qa_value(value: str | list[str] | None) -> str:
    """Normalize QA field value (string or list) to string."""
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _normalize_choices(choices: dict | None) -> dict[str, str]:
    """Normalize choices dict values to strings."""
    if not choices:
        return {}
    return {k: _normalize_qa_value(v) for k, v in choices.items()}


def _record_to_case(record: dict, dataset_name: str) -> Case:
    """Convert a MemBench record dict to a Case."""
    tid = record["tid"]
    sample_id = f"{dataset_name}::{tid}"
    conversation = _to_conversation(record["message_list"])

    qa = record["QA"]
    question = qa.get("question", "")
    choices = _normalize_choices(qa.get("choices"))
    answer = _normalize_qa_value(qa.get("answer"))
    ground_truth = _normalize_qa_value(qa.get("ground_truth"))

    return Case(
        id=sample_id,
        task_type="qa",
        inputs={
            "conversation": conversation,
            "question": question,
            "choices": choices,
        },
        expected_output=ground_truth or answer,
        metadata={
            "source_file": dataset_name,
            "tid": tid,
        },
    )


def load_membench(
    agent_type: str = "FirstAgent",
    limit: int | None = None,
    cache_dir: Path | str | None = None,
) -> list[Case]:
    """
    Load MemBench dataset.

    Args:
        agent_type: "FirstAgent" or "ThirdAgent"
        limit: Optional limit on number of samples
        cache_dir: Optional cache directory for git repository (defaults to ~/.cache/agentbench)

    Returns:
        List of Cases
    """
    if agent_type not in ("FirstAgent", "ThirdAgent"):
        raise ValueError(
            f"Invalid agent_type '{agent_type}'. Available: ['FirstAgent', 'ThirdAgent']"
        )

    cache_dir = Path.home() / ".cache" / "agentbench" if cache_dir is None else Path(cache_dir)
    repo_dir = cache_dir / "Membench"
    _ensure_membench_repo(repo_dir)

    source_dir = repo_dir / "MemData" / agent_type
    if not source_dir.exists():
        raise FileNotFoundError(f"MemBench data not found at {source_dir}")

    records_with_names: list[tuple[dict, str]] = []
    for json_path in sorted(source_dir.glob("*.json")):
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            records = data.get("events") or data.get("multi_agent") or []
            for record in records:
                records_with_names.append((record, json_path.stem))
                if limit is not None and len(records_with_names) >= limit:
                    break

            if limit is not None and len(records_with_names) >= limit:
                break
        except Exception as e:
            logger.warning(f"Failed to load MemBench file {json_path}: {e}")
            continue

    records_with_names.sort(key=lambda r: r[0]["tid"])

    if limit is not None:
        records_with_names = records_with_names[:limit]

    return [_record_to_case(record, dataset_name) for record, dataset_name in records_with_names]


def _ensure_membench_repo(repo_dir: Path) -> None:
    """Clone or update MemBench repository from GitHub."""
    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    if repo_dir.exists() and (repo_dir / ".git").exists():
        try:
            repo = Repo(repo_dir)
            repo.remotes.origin.fetch()
            repo.git.checkout(_MEMBENCH_GITHUB_BRANCH)
            repo.git.pull()
        except Exception as e:
            raise RuntimeError(f"Failed to update MemBench repository: {e}") from e
    else:
        try:
            Repo.clone_from(_MEMBENCH_GITHUB_REPO_URL, repo_dir, branch=_MEMBENCH_GITHUB_BRANCH)
        except Exception as e:
            raise RuntimeError(f"Failed to clone MemBench repository: {e}") from e
