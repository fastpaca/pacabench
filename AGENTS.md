# AI Agent Instructions for PacaBench (Rust Workspace)

> **Target Audience**: AI coding agents (Cursor, Copilot, Aider, etc.)

This repository is the **Rust rewrite** of PacaBench. These guidelines define how agents should work in this workspace, with a strong bias toward **type-first design**, **explicit error handling**, and **simple, composable code**.

See `CLAUDE.md` for a more detailed style and architecture guide.

---

## Critical: Code Quality Gates

Before marking any **Rust-related task** complete, you MUST run:

```bash
# Format all Rust code
cargo fmt --all

# Lint with clippy (no warnings allowed)
cargo clippy --workspace --all-targets --all-features -- -D warnings

# Run the full test suite
cargo test --workspace
```

- If `clippy` or `cargo test` fails, the task is **not complete**. Fix all issues first.
- CI (GitHub Actions) also runs `fmt`, `clippy`, and tests; keep the workspace clean locally.

For **docs-only** or configuration-only edits (e.g., `README.md`, `AGENTS.md`, `examples/*.yaml`), running `fmt/clippy/test` is recommended but not strictly required.

---

## Quick Reference

### Workspace Layout

```text
Cargo.toml               # Workspace root
crates/
  pacabench-core/        # Core benchmarking library
  pacabench-cli/         # CLI binary (pacabench)
examples/
  membench_qa_test/      # Example pacabench.yaml for quick QA runs
  smoke_test/            # Minimal smoke test config
runs/                    # (Created at runtime) benchmark outputs
```

### Core Commands

```bash
# Build everything
cargo build --workspace

# Run CLI help
cargo run -p pacabench-cli -- --help

# Quick smoke run using default pacabench.yaml in CWD
cargo run -p pacabench-cli -- run --limit 10

# Use an example config
cargo run -p pacabench-cli -- \
  --config examples/membench_qa_test/pacabench.yaml \
  run --limit 2

# Show runs
cargo run -p pacabench-cli -- show

# Retry failures from a run
cargo run -p pacabench-cli -- retry <run-id>

# Export results as JSON
cargo run -p pacabench-cli -- export <run-id> --format json
```

---

## Rust Code Style Rules (High-Level)

These rules are enforced socially and via `clippy` where possible. See `CLAUDE.md` for deeper rationale and examples.

### 1. Type-First Design

- Model domain concepts as **structs, enums, and newtypes** (e.g., `RunId(String)`, `CaseKey`, `NonEmpty<String>`).
- Avoid **stringly-typed** or bool-flag APIs for domain concepts.
- Use rich standard types (`Duration`, `PathBuf`, `NonZeroU*`, `Url`) instead of raw primitives where appropriate.
- Prefer borrowing (`&str`, `&[u8]`, `Cow<'a, str>`) over unnecessary allocation and cloning.

### 2. Failures as Data, Not Panics

- Use `Result<T, E>` and `Option<T>` for all fallible/optional operations; propagate errors with `?`.
- Library and core code should **not** use `unwrap`, `expect`, or `panic!` on recoverable failures.
- Allow `unwrap`/`expect` only in:
  - Tests and benchmarks.
  - Truly unreachable states with a clear, documented invariant (and even then prefer `unreachable!()`).
- Failures are **OK** as long as they are represented explicitly and pushed upstream in a `Result`.

### 3. Defaults: Allowed but Extremely Deliberate

- Using `Default` is fine, but only when the semantics are **obvious and safe** for every field.
- For domain/config types:
  - Prefer explicit constructors or builders (e.g., `Config::from_file(...)`, `ConfigOverrides::default()` with clear semantics).
  - Avoid `..Default::default()` when some fields are **required** for correctness.
- Never hide business-critical settings (timeouts, limits, feature flags) in implicit defaults.

### 4. Simplicity, Not Premature Generalisation

- Avoid generic abstractions, traits, or type gymnastics until there are **at least two real use-cases**.
- Prefer straightforward functions and concrete types over “framework-style” layers.
- Refactor duplication only when the new abstraction is:
  - Simple,
  - Clearly named, and
  - Local to the domain where it’s used.

### 5. Concurrency and Locks

- Prefer **clear ownership** and **message passing** over shared mutable state.
- Avoid reaching for `Mutex`/`RwLock`/`parking_lot` as a default. If you feel the need to add a lock, first:
  - Ask if you can pass owned data into tasks instead of sharing.
  - Consider a single owner task with channels for requests/responses.
  - Consider restructuring the concurrency model (e.g., workers + queue).
- Locks are allowed but should be **rare, narrow in scope, and well-justified**.
- In async code, never block on sync I/O; use async I/O or `spawn_blocking` for heavy work.

### 6. Error Handling and Logging

- Define dedicated error types per domain (`ConfigError`, `PersistenceError`, etc.), implemented with `thiserror` or manual enums.
- Use typed errors inside `pacabench-core`; use `anyhow` primarily at the CLI/binary boundary for flexible reporting.
- Add context when crossing layers (e.g., `"loading run metadata"`, `"parsing pacabench.yaml"`).
- Avoid “log and swallow” patterns. Either:
  - Handle the error fully and return a valid result, or
  - Return an error upstream.

---

## Architecture Notes (Rust Rewrite)

- **Library-first design**:
  - `pacabench-core` exposes `Config`, `Benchmark`, `Event`, `Command`, `CaseResult`, and aggregated metrics.
  - `pacabench-cli` is a **thin wrapper** that:
    - Parses CLI options (via `clap`),
    - Loads configuration,
    - Calls into `Benchmark`,
    - Renders progress and metrics.

- **Proxy and metrics**:
  - All LLM traffic should flow through the proxy implemented in `pacabench-core::proxy`.
  - Metrics (`AggregatedMetrics`) must continue to include:
    - Duration percentiles (p50/p95),
    - LLM latency (avg/p50/p95),
    - Token counts (input/output/judge/cached),
    - Attempt counts and failure counts.

- **Results and persistence**:
  - Per-case and aggregated results are persisted via `pacabench-core::persistence`.
  - CLI commands (`show`, `retry`, `export`) must keep these formats stable or evolve them in a backwards-compatible way.

---

## Non-Negotiables

1. **Always** run `cargo fmt`, `cargo clippy` (with `-D warnings`), and `cargo test` before merging Rust code changes.
2. **Never** remove latency, token, or cost metrics from CLI or JSON/Markdown outputs.
3. **Never** introduce `unwrap`/`expect`/`panic!` in production paths; use `Result`/`Option` and propagate errors.
4. **Never** add unnecessary abstraction layers or premature generalisation; keep code as simple as the domain allows.
5. **Avoid** new locks (`Mutex`/`RwLock`/`parking_lot` types) unless the design has been considered and documented; prefer ownership and channels.
6. **Always** model important domain concepts as types, not loose strings or integers.

Follow these rules, and use `CLAUDE.md` for deeper Rust-specific gotchas, anti-patterns, and examples when in doubt.

