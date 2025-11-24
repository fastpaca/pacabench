# Contributing to PacaBench

Thank you for your interest in contributing to PacaBench! We welcome contributions from the community to help make this the standard harness for agentic evaluation.

## Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally.
3.  **Install dependencies**:
    ```bash
    # We recommend using uv for dependency management
    pip install uv
    uv sync --all-extras
    ```
4.  **Create a branch** for your feature or fix.

## Development Workflow

### Code Style

We enforce strict code quality gates using `ruff`. Before submitting a PR, you **must** run:

```bash
# Fix linting issues
uv run ruff check pacabench/ --fix

# Format code
uv run ruff format pacabench/

# Verify everything is clean
uv run ruff check pacabench/
```

### Running Tests

We use `pytest` for testing (if tests are available) and the harness itself for smoke testing.

To run a quick smoke test of the harness:

```bash
cd examples/smoke_test && uv run pacabench
```

## Pull Request Process

1. Ensure all checks pass locally (`ruff` and smoke tests).
2. Open a Pull Request against the `main` branch.
3. Provide a clear description of the problem you are solving and your solution.
4. If you are adding a new feature, include a test case or verification steps.
5. **If you are using AI** ensure code quality is high before you submit.

## Reporting Issues

Please use the GitHub Issue Tracker to report bugs or request features. Include as much detail as possible, such as:
-   PacaBench version
-   Configuration file (`pacabench.yaml`)
-   Steps to reproduce
-   Logs or error messages

## License

By contributing, you agree that your contributions will be licensed under its [Apache 2.0 License](LICENSE).

