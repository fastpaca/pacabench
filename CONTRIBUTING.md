# Contributing to PacaBench

Thank you for your interest in contributing to PacaBench! We welcome contributions from the community to help make this the standard harness for agentic evaluation.

## Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally.
3.  **Install Rust** via [rustup](https://rustup.rs/):
    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    ```
4.  **Build the project**:
    ```bash
    cargo build --workspace
    ```
5.  **Create a branch** for your feature or fix.

## Development Workflow

### Code Quality

Before submitting a PR, you **must** run:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
```

All three commands must pass with no errors or warnings.

### Running Smoke Tests

```bash
cd examples/smoke_test && cargo run -p pacabench-cli
```

### Project Structure

- `crates/pacabench-core/` - Core library with benchmark engine, persistence, and proxy
- `crates/pacabench-cli/` - CLI binary (thin wrapper around core)
- `examples/` - Example benchmark configurations

## Pull Request Process

1. Ensure all checks pass locally.
2. Open a Pull Request against the `main` branch.
3. Provide a clear description of the problem and your solution.
4. If adding a new feature, include test cases.
5. **If using AI**, ensure code quality is high before you submit.

## Reporting Issues

Please use the GitHub Issue Tracker. Include:
- PacaBench version
- Configuration file (`pacabench.yaml`)
- Steps to reproduce
- Logs or error messages

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).
