# Claude Instructions for PacaBench (Rust Workspace)

This document provides **Rust-specific** guidance for Claude (and other LLMs) working on the PacaBench codebase.

The goals:
- Prefer **type-first, explicit design** over clever abstractions.
- Make failures **visible and explicit** (via `Result`), not hidden panics.
- Keep concurrency **simple and well-structured**, avoiding unnecessary locks.

`AGENTS.md` covers process and commands; this file focuses on **style, patterns, and anti-patterns**.

---

## Quality Tooling (Always Run for Rust Changes)

For any task that touches Rust code, you MUST ensure the workspace is clean:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
```

- Do not consider a Rust task “done” until `fmt`, `clippy`, and tests pass.
- For docs-only edits, running them is recommended but not mandatory.

---

## Design Philosophy

1. **Type-first modeling** – encode domain concepts (runs, agents, datasets, metrics) as well-typed structs/enums/newtypes.
2. **Failures as data** – represent every failure as a `Result<T, E>` or `Option<T>`, never as a hidden panic.
3. **Simplicity over abstraction** – avoid generic traits and frameworks until there are multiple real use-cases.
4. **Minimal, deliberate concurrency** – prefer clear ownership and message-passing; locks are a last resort.
5. **Defaults are explicit** – using `Default` is fine, but only when the semantics are obvious and carefully designed.

---

## Rust Rules: Do / Don’t

### Types and Modeling

**Do:**
- Introduce newtypes and enums for domain concepts:
  - `RunId(String)`, `DatasetName(String)`, `AgentName(String)`.
  - `enum RunStatus { Pending, Running, Completed, Failed }`.
- Use richer types from std/third-party crates:
  - `std::time::Duration`, `std::path::PathBuf`, `std::num::NonZeroU64`.
  - `serde_json::Value` only at boundaries where the shape is truly dynamic.
- Prefer borrowing (`&str`, `&[u8]`, `Cow<'a, str>`) over unnecessary `String`/`Vec<u8>` ownership.

**Don’t:**
- Don’t use “stringly typed” identifiers, states, or enums (`String`/`&str` with magic values).
- Don’t pass around `bool` flags where an enum would make intent clear.
- Don’t expose invariants as comments; enforce them in constructors and types.

---

### Error Handling and `Result`

**Do:**
- Use `Result<T, E>` everywhere an operation can fail.
- Define focused error enums per domain (e.g., `ConfigError`, `PersistenceError`) with `thiserror`:
  ```rust
  #[derive(Debug, thiserror::Error)]
  pub enum ConfigError {
      #[error("failed to read config file {path}: {source}")]
      Io {
          path: PathBuf,
          #[source]
          source: std::io::Error,
      },
      #[error("invalid benchmark config: {0}")]
      Invalid(String),
  }
  ```
- Use `anyhow::Result` only at the **outermost layers** (main/CLI) for user-facing error reporting.
- Add context when crossing layers (e.g., `"loading run summaries"`, `"parsing pacabench.yaml"`).

**Don’t (Anti-Patterns):**
- Don’t use `unwrap`, `expect`, or `panic!` in production paths.
- Don’t convert typed errors to `String` or `Box<dyn std::error::Error>` in core logic.
- Don’t log an error and then return `Ok(())` just to keep going; either handle or propagate.
- Don’t use `Result<(), ()>` as a generic error channel; define a concrete error type.

**Allowed Exceptions:**
- Tests and benchmarks may use `unwrap`/`expect`, but with meaningful messages.
- Truly unrecoverable issues at the very edge of the binary may `panic!`, though a typed error is still preferred.

---

### Defaults and Configuration

**Do:**
- Use `Default` when there is a **clear, safe, and obvious** default (e.g., empty collections, simple option structs).
- Use `ConfigOverrides::default()` and similar types where the default truly means “no override”.
- Provide constructors/builders that enforce invariants:
  ```rust
  impl Config {
      pub fn from_file(path: &Path, overrides: ConfigOverrides) -> Result<Self, ConfigError> {
          // parse + validate
      }
  }
  ```

**Don’t:**
- Don’t derive `Default` for important domain or config types when fields must be consciously set.
- Don’t rely on `..Default::default()` to silently fill in crucial values like timeouts, limits, or feature flags.
- Don’t use `unwrap_or_default()` on important domain types; handle the case explicitly.

---

### Concurrency, Async, and Locks

**Do:**
- Prefer **structured concurrency**:
  - A small number of clearly-owned tasks (workers) managed by a central controller (e.g., `Benchmark` + workers).
  - Channels (`async_channel`, `tokio::sync::mpsc`) for requests and events.
- Keep async boundaries clear:
  - Avoid mixing sync and async I/O in the same function.
  - Use `tokio::task::spawn_blocking` for CPU-heavy or blocking operations if needed.
- Ensure that anything crossing threads is `Send + Sync`; let the type system enforce this.

**Don’t (Anti-Patterns):**
- Don’t introduce `Mutex`/`RwLock` (including `parking_lot` variants) as a default way to share state.
  - If you feel tempted to add a lock, first reconsider the concurrency design:
    - Can one task own the state?
    - Can you send messages instead of sharing?
- Don’t create “god” shared states like `Arc<Mutex<Vec<_>>>` that everything mutates.
- Don’t spawn detached tasks (`tokio::spawn`) without a clear owner or shutdown path.
- Don’t block inside async code with sync I/O or long computations.

Locks are **allowed**, but they must be:
- Narrow in scope.
- Justified in comments.
- Avoidable only at disproportionate cost.

---

### Simplicity and Abstractions

**Do:**
- Start with concrete, straightforward implementations.
- Factor out common code only when duplication is provably painful **and** the abstraction is easy to understand.
- Keep traits small and focused; prefer composition over inheritance-style hierarchies.

**Don’t:**
- Don’t introduce generic traits and blanket impls “just in case”.
- Don’t wrap simple types in multiple layers of indirection, builders, or generics.
- Don’t overuse macros; use them when they genuinely reduce boilerplate with clear semantics.

---

### Project-Specific Rules (PacaBench)

**Metrics & Proxy:**
- Always maintain complete metrics:
  - Accuracy/precision/recall and failed case counts.
  - Duration percentiles (p50/p95).
  - LLM latency (avg/p50/p95).
  - Token counts (input, output, judge, cached).
  - Attempts (avg/max).
- Never remove or silently downgrade latency/token/cost metrics from CLI or exported formats (JSON/Markdown).
- Ensure all LLM calls within the benchmark path go through the proxy layer in `pacabench-core::proxy` so metrics stay correct.

**CLI (`pacabench-cli`):**
- Keep the CLI a **thin wrapper**:
  - Parse arguments with `clap`.
  - Construct `Config` / `ConfigOverrides`.
  - Call into `Benchmark`.
  - Render progress and metrics (no complex business logic).
- Avoid duplicating logic that already exists in `pacabench-core`.

**Persistence and Schemas:**
- Use `serde` for stable on-disk formats in `pacabench-core::persistence`.
- When changing result formats:
  - Prefer additive, backwards-compatible changes.
  - Update both CLI display and export code in sync.

---

### Testing and Documentation

**Do:**
- Add unit tests alongside modules in `crates/pacabench-core/src` for core logic.
- Add integration tests to `crates/pacabench-cli/tests` for end-to-end CLI behavior.
- Use `#[tokio::test]` for async scenarios.
- Document public APIs with `///` comments, explaining:
  - Purpose.
  - Invariants.
  - Errors and panics (if any).

**Don’t:**
- Don’t rely solely on CLI smoke tests for core behavior; important logic should have unit tests.
- Don’t leave public functions undocumented in the core library.

---

## Hard Anti-Patterns (Treat as Errors)

Claude should treat the following as **red flags** and avoid generating them:

1. `unwrap` / `expect` / `panic!` in non-test Rust code paths.
2. New `Mutex`/`RwLock`/`parking_lot` locks without an explicit design reason and minimal scope.
3. Deriving or using `Default` for non-trivial domain or config types where “default” is not obvious and safe.
4. Stringly-typed identifiers, states, and flags instead of newtypes/enums.
5. Logging and then ignoring errors instead of handling or propagating them.
6. Overly defensive programming (re-checking invariants everywhere instead of enforcing them at construction).
7. Premature generalisation via generic traits, lifetimes, or macros that do not clearly simplify real code.
8. Removing or weakening metrics (especially latency, tokens, and cost) in CLI output or exported formats.

If you are unsure about a trade-off, prefer:
- **Concrete over generic**,
- **Typed over stringly**,
- **Result over panic**,
- **Simple concurrency over “clever” locking**.

