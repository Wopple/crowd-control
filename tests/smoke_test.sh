#!/bin/bash
# Smoke test for crowd-control package installation.
# Run in a clean virtualenv. Does not require Ollama, Claude, or any API keys.
# Exit on first failure for commands that must succeed.
set -euo pipefail

FAILURES=0

# Helper: assert a command exits 0
assert_ok() {
    echo "  [PASS] $1"
}

# Helper: assert a command fails gracefully (expected message in output)
assert_graceful_fail() {
    local cmd="$1"
    local expected_msg="$2"
    local output
    output=$(eval "$cmd" 2>&1) || true
    if echo "$output" | grep -qi "$expected_msg"; then
        echo "  [PASS] $cmd → graceful error"
    else
        echo "  [FAIL] $cmd → expected '$expected_msg' in output, got: $output"
        FAILURES=$((FAILURES + 1))
    fi
}

echo "=== Install ==="
pip install .

echo "=== CLI entry point (must succeed) ==="
crowd-control --version
assert_ok "crowd-control --version"
crowd-control --help > /dev/null
assert_ok "crowd-control --help"

echo "=== Status without DB (graceful error) ==="
assert_graceful_fail "crowd-control status" "not initialized"

echo "=== Setup help (must succeed) ==="
crowd-control setup --help > /dev/null
assert_ok "crowd-control setup --help"

echo "=== List without DB (graceful error) ==="
assert_graceful_fail "crowd-control list" "not available\|No learnings"

echo "=== Search without embedder (graceful error) ==="
assert_graceful_fail "crowd-control search 'test query'" "embedding\|provider\|not available"

echo "=== Export without DB (graceful error) ==="
assert_graceful_fail "crowd-control export" "not available\|not initialized"

echo "=== Config loading (must succeed) ==="
python -c "from crowd_control.config import load_config; c = load_config(); print(f'OK: {c.storage_dir}')"
assert_ok "config import"

echo "=== Module imports (must succeed) ==="
python -c "from crowd_control.server import create_server; print('Server factory OK')"
python -c "from crowd_control.hooks import handle_session_end_hook; print('Hooks OK')"
python -c "from crowd_control.worker import process_queue; print('Worker OK')"
assert_ok "module imports"

echo ""
if [ "$FAILURES" -gt 0 ]; then
    echo "=== $FAILURES smoke test(s) FAILED ==="
    exit 1
else
    echo "=== All smoke tests passed ==="
fi
