# Offline smoke test

Contributor and CI check. Two local cases, an echo agent, and exact-match scoring. No API key and no network.

The Quick Start is [`examples/quickstart`](../quickstart/): two real model calls, with the metrics proxy on.

```bash
pacabench run
```

The agent reverses each question (`hello` → `olleh`). Both cases should pass in a few seconds.

When stdout is not a terminal, pass `--no-tui`:

```bash
pacabench run --no-tui
```

Requires `python` on `PATH` (Python 3). The agent uses the standard library only.
