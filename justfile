# Recipes use bash (shebang recipes + the _ci launcher), so run them in bash.
set shell := ["bash", "-cu"]

# List all available recipes
default:
    @just --list

# Per-user scratch dir. Torch-TensorRT's engine/timing cache lives under
# $TMPDIR/torch_tensorrt_engine_cache; on shared hosts the default /tmp path is
# created by whoever runs first and then fails for everyone else with a
# PermissionError. Giving each user their own TMPDIR sidesteps that entirely.
export TMPDIR := env_var_or_default("TMPDIR", "/tmp/torch_tensorrt_" + env_var_or_default("USER", "local"))

# pytest xdist parallelism for the parallel suites. Defaults to 4: these suites
# build TensorRT engines, which are GPU-memory-heavy, and a single local GPU
# OOMs (CUDA out-of-memory + segfaulting workers) well before it runs out of
# CPU cores — so `-n auto` is the wrong default here. CI runs more on dedicated
# GPU runners. Raise it if your GPU has headroom:  just jobs=8 lane full
jobs := "4"

# Which backend variant to run: standard | rtx. This selects the test SELECTION
# (RTX drops --dist on converters, runs the whole partitioning dir, etc.) — it
# does NOT switch the installed build. To actually run RTX locally you must have
# a torch-tensorrt built with USE_TRT_RTX=1 installed in the .venv.
#   just variant=rtx lane fast
variant := "standard"

# Launcher + env for the suite recipes. Suite definitions live in the manifest
# (tests/ci/suites.py) — the SAME data CI uses — so local and CI cannot drift.
# We only set environment policy here: PYTHON runs pytest against the already-built
# .venv (uv --no-sync, no rebuild), TRT_JOBS feeds xdist, and reruns are off
# locally (the rerunfailures plugin may be absent and you want to SEE flakes).
_ci := 'mkdir -p "$TMPDIR" && PYTHON="uv run --no-sync python" TRT_JOBS="' + jobs + '" TRT_PYTEST_RERUNS=0 uv run --no-sync python -m tests.ci'

# ── Testing ───────────────────────────────────────────────────────────────────

# Run pytest in the uv-managed env (honors pyproject addopts). Pass any args:
#   just test tests/py/dynamo/conversion/
#   just test -k test_foo -x tests/py/dynamo/runtime/
test *args:
    @mkdir -p "$TMPDIR"
    uv run --no-sync pytest {{args}}

# ── Suite manifest (tests/ci) — the single source of truth ──────────────────────
#
# CI and these recipes both call `python -m tests.ci`, so what runs locally is
# exactly what runs in CI. See TESTING_AND_CI_DESIGN.md.

# Validate the suite manifest (unique names/junit paths, valid setup steps)
doctor:
    uv run --no-sync python -m tests.ci doctor

# List every suite with its tier, lanes, and variants
suites:
    uv run --no-sync python -m tests.ci list

# Run ONE suite exactly as CI runs it (uses the {{variant}} backend). Args after `--`:
#   just suite dynamo-runtime -- -k test_foo -x        just variant=rtx suite dynamo-converters
suite name *args:
    {{_ci}} run {{name}} --variant {{variant}} {{args}}

# Run every suite in a LANE (fast|full|nightly), continuing past failures:
#   just lane fast        just variant=rtx lane fast
lane name *args:
    {{_ci}} run-lane --lane {{name}} --variant {{variant}} {{args}}

# Run a lane past failures, then print one consolidated report (--agent for Claude):
#   just report fast --agent
report lane *summary_args:
    #!/usr/bin/env bash
    set -uo pipefail
    mkdir -p "$TMPDIR"
    results="${RUNNER_TEST_RESULTS_DIR:-$TMPDIR/trt_test_results}"
    rm -f "$results"/*.xml 2>/dev/null || true
    export PYTHON="uv run --no-sync python" TRT_JOBS="{{jobs}}" TRT_PYTEST_RERUNS=0
    uv run --no-sync python -m tests.ci run-lane --lane {{lane}} --variant {{variant}} || true
    uv run --no-sync python tests/py/utils/junit_summary.py "$results" {{summary_args}}

# Re-print the LAST run's consolidated report without re-running (--agent for Claude)
summary *args:
    uv run --no-sync python tests/py/utils/junit_summary.py "${RUNNER_TEST_RESULTS_DIR:-$TMPDIR/trt_test_results}" {{args}}

# Added without a rebuild via `uv pip install --group` (sidesteps the
# test-ext↔quantization `uv sync` lockfile conflict). Run before `just lane full`.
# Install optional test deps so model/kernels/quantization/executorch suites run
install-test-ext:
    uv pip install --group test-ext --group kernels --group quantization
    uv pip install pyyaml "executorch>=1.3.1"

# ── Linting ───────────────────────────────────────────────────────────────────

# Run all pre-commit hooks across the repo (matches the linter CI job)
lint:
    uv run --no-sync --with pre-commit pre-commit run --all-files

# Run pre-commit only on files changed vs origin/main (fast pre-push check)
lint-changed:
    uv run --no-sync --with pre-commit pre-commit run --from-ref origin/main --to-ref HEAD

# ── Docs ──────────────────────────────────────────────────────────────────────

# Build HTML documentation (manages deps via uv)
docs:
    cd docsrc && uv run --group docs --prerelease allow make html

# Clean docs build artifacts (does not touch the published docs/ directory)
docs-clean:
    rm -rf docsrc/_build docsrc/_cpp_api docsrc/_py_api docsrc/_tmp docsrc/tutorials/_rendered_examples

# Serve the already-built docs locally
docs-serve port="3000":
    python3 -m http.server {{port}} --directory docs

# Build docs then serve them
docs-build-serve port="3000": docs
    python3 -m http.server {{port}} --directory docs
