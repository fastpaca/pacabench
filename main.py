from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

from memharness import (
    DatasetConfig,
    get_dataset,
    list_answerers,
    list_datasets,
    load_dataset,
    build_answerer,
    evaluate_samples,
)


def datasets_command(args: argparse.Namespace) -> None:
    specs = list_datasets()
    for spec in specs:
        print(f"{spec.name}: {spec.description} (splits: {', '.join(spec.available_splits)})")


def answerers_command(args: argparse.Namespace) -> None:
    specs = list_answerers()
    for spec in specs:
        print(f"{spec.name}: {spec.description}")


def run_command(args: argparse.Namespace) -> None:
    dataset_spec = get_dataset(args.dataset)
    dataset_config = DatasetConfig(
        root=Path(args.data_root).expanduser().resolve() if args.data_root else dataset_spec.default_root,
        split=args.split,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        include=args.include,
        limit=args.dataset_limit,
    )

    samples = load_dataset(args.dataset, dataset_config)
    if not samples:
        raise SystemExit("No samples found for the requested configuration.")

    answerer = build_answerer(
        args.answerer,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        run_dir = Path("runs") / f"{args.dataset}-{timestamp}"

    if run_dir.exists():
        raise SystemExit(f"Run directory {run_dir} already exists.")
    run_dir.mkdir(parents=True)

    run_config = {
        "dataset": args.dataset,
        "dataset_config": {
            "root": str(dataset_config.root),
            "split": dataset_config.split,
            "eval_ratio": dataset_config.eval_ratio,
            "seed": dataset_config.seed,
            "include": dataset_config.include,
            "dataset_limit": dataset_config.limit,
        },
        "answerer": {
            "name": args.answerer,
            "provider": args.provider,
            "model": args.model,
            "base_url": args.base_url,
        },
        "evaluation": {
            "limit": args.eval_limit,
            "shuffle": args.shuffle,
            "shuffle_seed": args.shuffle_seed,
        },
    }
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as config_file:
        json.dump(run_config, config_file, indent=2)

    metrics = evaluate_samples(
        samples,
        answerer,
        run_dir,
        limit=args.eval_limit,
        shuffle=args.shuffle,
        shuffle_seed=args.shuffle_seed,
    )
    print(json.dumps(metrics, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="memharness: Memory Evaluation Harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    datasets_parser = subparsers.add_parser("datasets", help="List available datasets.")
    datasets_parser.set_defaults(func=datasets_command)

    answerers_parser = subparsers.add_parser("answerers", help="List available answerers.")
    answerers_parser.set_defaults(func=answerers_command)

    run_parser = subparsers.add_parser("run", help="Execute an evaluation run.")
    run_parser.add_argument("--dataset", required=True, help="Dataset identifier, e.g., memharness-first-agent.")
    run_parser.add_argument("--split", default="eval", help="Dataset split (default: eval).")
    run_parser.add_argument("--data-root", help="Override dataset root directory.")
    run_parser.add_argument("--eval-ratio", type=float, default=0.8, help="Eval split ratio when applicable.")
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting and sampling.")
    run_parser.add_argument("--include", nargs="*", help="Optional filters for dataset subsets.")
    run_parser.add_argument("--dataset-limit", type=int, help="Limit number of dataset samples before eval.")

    run_parser.add_argument("--answerer", default="pydantic-agent", help="Answerer identifier.")
    run_parser.add_argument("--provider", default="openai", help="Model provider for the pydantic-agent answerer.")
    run_parser.add_argument("--model", default="lm-studio", help="Model name served by the provider.")
    run_parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1", help="Endpoint base URL.")
    run_parser.add_argument("--api-key", help="API key/token for the provider (defaults to placeholder).")

    run_parser.add_argument("--eval-limit", type=int, help="Limit number of evaluation samples.")
    run_parser.add_argument("--shuffle", action="store_true", help="Shuffle samples before evaluation.")
    run_parser.add_argument("--shuffle-seed", type=int, default=42, help="Seed used when shuffling.")
    run_parser.add_argument("--run-dir", help="Directory to store run artifacts (default: runs/<dataset>-<timestamp>).")

    run_parser.set_defaults(func=run_command)

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
