#!/bin/bash
# Record all demo GIFs for the README
#
# Usage:
#   cd examples/membench_qa_test
#   ../../docs/scripts/record-readme-demos.sh
#   ../../docs/scripts/record-readme-demos.sh kill-resume
#
# Prerequisites:
#   asciinema and agg on PATH (brew install asciinema agg, or the upstream release binaries)
#   cargo build --release (or the script will do it)
#
# kill-resume records docs/images/kill-resume.gif from examples/smoke_test.
# Ctrl+C finalizes a run as aborted, so the recorder SIGKILLs the process.
# The same `pacabench run --run-id` then resumes, and completed cases stay done.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TUI_DEMO="$SCRIPT_DIR/tui-demo"
OUTPUT_DIR="$REPO_ROOT/docs/images"
DEMO_COLS=140
DEMO_ROWS=40
DEMO_THEME="dracula"
DEMO_FONT_SIZE=10
DEMO_ARGS=(--cols "$DEMO_COLS" --rows "$DEMO_ROWS" --theme "$DEMO_THEME" --font-size "$DEMO_FONT_SIZE")
DEMO_FILTER="${1:-all}"
# agg --font-family does not fall back. Liberation Mono is the face used for
# the committed kill-resume GIF. Override when re-recording elsewhere.
KILL_RESUME_FONT="${KILL_RESUME_FONT:-Liberation Mono}"

should_record() {
    [[ "$DEMO_FILTER" == "all" || "$DEMO_FILTER" == "$1" ]]
}

if ! should_record kill-resume || [[ "$DEMO_FILTER" == "all" ]]; then
    if [[ ! -f "pacabench.yaml" ]]; then
        echo "Error: Run this from an example directory (e.g., examples/membench_qa_test)"
        exit 1
    fi
fi

# Build pacabench
echo "Building pacabench..."
cargo build --manifest-path "$REPO_ROOT/Cargo.toml" --release --bin pacabench -q
PACABENCH="$REPO_ROOT/target/release/pacabench"

# Add to PATH so tui-demo can find it
export PATH="$(dirname "$PACABENCH"):$PATH"

echo ""
echo "Recording README demos..."
echo "Output: $OUTPUT_DIR"
echo ""

# Ensure colors are enabled in recorded output.
unset NO_COLOR

# Demo 1: Running a benchmark with --limit
if should_record simple-run; then
echo "==> Recording: simple-run.gif (pacabench run --limit 15)"
"$TUI_DEMO" \
    -o "$OUTPUT_DIR/simple-run.gif" \
    "${DEMO_ARGS[@]}" \
    --pause 3 \
    -- "pacabench run --limit 15"

echo ""
fi

# Demo 2: List all runs
if should_record simple-show; then
echo "==> Recording: simple-show.gif (pacabench show)"
"$TUI_DEMO" \
    -o "$OUTPUT_DIR/simple-show.gif" \
    "${DEMO_ARGS[@]}" \
    --pause 2.5 \
    -- "pacabench show"

echo ""
fi

# Demo 3: Show specific run details
if should_record simple-show-detail; then
echo "==> Recording: simple-show-detail.gif (pacabench show <id>)"

# Get the most recent completed run ID
LATEST_RUN=$($PACABENCH show 2>/dev/null | grep completed | head -1 | awk '{print $1}')
if [[ -z "$LATEST_RUN" ]]; then
    echo "Error: No completed runs found. Run 'pacabench run --limit 15' first."
    exit 1
fi

# Use the last 8 chars of the run id (nanoid can include hyphens)
SHORT_ID="${LATEST_RUN: -8}"

"$TUI_DEMO" \
    -o "$OUTPUT_DIR/simple-show-detail.gif" \
    "${DEMO_ARGS[@]}" \
    --pause 2.5 \
    -- "pacabench show $SHORT_ID"

echo ""
fi

# Demo 4: Kill a slow run, then resume the same id.
# SIGKILL, not SIGINT: Ctrl+C marks the run aborted and does not continue
# cases that never started.
if should_record kill-resume; then
KILL_RESUME_DIR="$REPO_ROOT/examples/smoke_test"
KILL_RESUME_ID="kill-resume-demo"
echo "==> Recording: kill-resume.gif (SIGKILL, then the same --run-id)"
rm -rf "$KILL_RESUME_DIR/runs/$KILL_RESUME_ID"
(
    cd "$KILL_RESUME_DIR"
    "$TUI_DEMO" \
        -o "$OUTPUT_DIR/kill-resume.gif" \
        "${DEMO_ARGS[@]}" \
        --font-family "$KILL_RESUME_FONT" \
        --typing-speed 0.02 \
        --pause 1.5 \
        --kill-after 3.4 \
        -- "pacabench -c pacabench.kill-resume.yaml run --run-id $KILL_RESUME_ID --no-tui" \
           "pacabench -c pacabench.kill-resume.yaml run --run-id $KILL_RESUME_ID --no-tui"
)
echo ""
fi

echo "Done! Generated:"
if should_record simple-run; then
echo "  $OUTPUT_DIR/simple-run.gif"
fi
if should_record simple-show; then
echo "  $OUTPUT_DIR/simple-show.gif"
fi
if should_record simple-show-detail; then
echo "  $OUTPUT_DIR/simple-show-detail.gif"
fi
if should_record kill-resume; then
echo "  $OUTPUT_DIR/kill-resume.gif"
fi
