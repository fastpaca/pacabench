# PacaBench

Agent source of truth. `pacabench-core` owns the domain. `pacabench-cli` is a thin clap/TUI wrapper.

## Quality gates

A Rust change is done only when these pass. CI runs the same commands.

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo nextest run --workspace
cargo test --doc --workspace
```

`cargo test --doc` stays in the gate. `pacabench-core` has a doctest, and nextest does not run doctests. Docs-only edits (`README.md`, this file, `examples/`) do not need the gates.

MSRV is **1.85.0** (`workspace.package.rust-version`), the edition-2024 floor this lockfile builds on. `cargo +1.85.0 check --workspace --locked` must pass. CI checks that on its own job. Clippy and nextest run on stable.

rustfmt and clippy use tool defaults. Enforcement is the commands above. There is no extra `.clippy.toml` or `rustfmt.toml`.

## Errors

- `pacabench-core`: typed errors (`thiserror`, or an existing enum such as `PacabenchError`). Propagate with `?`.
- `pacabench-cli`: `anyhow` only at the binary boundary.
- No `unwrap`, `expect`, or `panic!` on production paths. Tests may use them.

## Process isolation

- One agent subprocess per runner.
- On timeout or stop, kill the process tree. Unix: the process group. Windows: best-effort direct-child kill until Job Objects exist.
- After a timeout, restart the runner so the next case is a fresh process.

## Persistence

- Case results are append-only JSONL.
- Metadata writes are atomic: temp file in the same directory, `fsync`, then rename over the destination.

## Retry

- Exponential backoff with jitter.
- Retry only retryable failures.
- The policy lives in one place (`RetryPolicy`).

## Module boundaries

- Domain concepts are types (newtypes, enums), not stringly-typed ids, states, or flags.
- No new `Mutex`, `RwLock`, or `parking_lot` lock unless a comment says why a single owner or a channel will not work. Keep the lock narrow.
- Keep metrics complete in CLI and JSON/Markdown export: duration p50/p95, LLM latency avg/p50/p95, tokens (input, output, judge, cached), attempts, failure counts.
- LLM calls on the benchmark path go through `pacabench-core::proxy`.

## Non-goals

- No product or marketing scope.
- Do not break the public CLI or library API in a drive-by cleanup.
