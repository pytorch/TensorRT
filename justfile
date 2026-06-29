# Recipes source tests/py/utils/ci_helpers.sh (a bash library), so run them in bash.
set shell := ["bash", "-cu"]

# List all available recipes
default:
    @just --list

# Per-user scratch dir. Torch-TensorRT's engine/timing cache lives under
# $TMPDIR/torch_tensorrt_engine_cache; on shared hosts the default /tmp path is
# created by whoever runs first and then fails for everyone else with a
# PermissionError. Giving each user their own TMPDIR sidesteps that entirely.
export TMPDIR := env_var_or_default("TMPDIR", "/tmp/torch_tensorrt_" + env_var_or_default("USER", "local"))

# pytest xdist parallelism for the parallel suites. Defaults to 2: these tiers
# build TensorRT engines, which are GPU-memory-heavy, and a single local GPU
# OOMs (CUDA out-of-memory + segfaulting workers) well before it runs out of
# CPU cores — so `-n auto` is the wrong default here. CI runs 8 on dedicated
# GPU runners. Raise it if your GPU has headroom:  just jobs=8 l0
jobs := "4"

# Launcher + env shared by every tier recipe. The tier definitions themselves
# live in tests/py/utils/ci_helpers.sh — the SAME functions CI calls — so there is a
# single source of truth for what each tier runs. We only set environment
# policy here: PYTHON runs pytest against the already-built .venv (uv --no-sync,
# no rebuild) and TRT_JOBS feeds the parallel suites.
_tier := 'mkdir -p "$TMPDIR" && source tests/py/utils/ci_helpers.sh && export PYTHON="uv run --no-sync python" TRT_JOBS="' + jobs + '" TRT_PYTEST_RERUNS=0 &&'

# ── Testing ───────────────────────────────────────────────────────────────────

# Run pytest in the uv-managed env (honors pyproject addopts). Pass any args:
#   just test tests/py/dynamo/conversion/
#   just test -k test_foo -x tests/py/dynamo/runtime/
test *args:
    @mkdir -p "$TMPDIR"
    uv run --no-sync pytest {{args}}

# ── CI tier reproduction ──────────────────────────────────────────────────────
#
# Run exactly what a CI tier runs, before pushing. Each recipe calls the tier
# function from tests/py/utils/ci_helpers.sh — the same one .github/workflows/
# _linux-x86_64-core.yml invokes — so local and CI cannot drift. Extra args are
# forwarded to pytest, e.g. `just tests-l0-core -x -k test_foo`.
# (Standard-TensorRT scope; export USE_TRT_RTX=true before running for RTX.)

# Full L0 smoke tier
tests-l0: tests-l0-converter tests-l0-core tests-l0-py-core tests-l0-torchscript

tests-l0-converter *args:
    {{_tier}} trt_tier_l0_converter {{args}}

tests-l0-core *args:
    {{_tier}} trt_tier_l0_core {{args}}

tests-l0-py-core *args:
    {{_tier}} trt_tier_l0_py_core {{args}}

tests-l0-torchscript *args:
    {{_tier}} trt_tier_l0_torchscript {{args}}

# Full L1 tier
tests-l1: tests-l1-dynamo-core tests-l1-dynamo-compile tests-l1-torch-compile tests-l1-torchscript

tests-l1-dynamo-core *args:
    {{_tier}} trt_tier_l1_dynamo_core {{args}}

tests-l1-dynamo-compile *args:
    {{_tier}} trt_tier_l1_dynamo_compile {{args}}

tests-l1-torch-compile *args:
    {{_tier}} trt_tier_l1_torch_compile {{args}}

tests-l1-torchscript *args:
    {{_tier}} trt_tier_l1_torchscript {{args}}

# Pulls every optional test-dep group so the model / kernels / quantization /
# executorch suites run instead of skipping. Uses `uv pip install` (not `uv
# sync`) so it ADDS the deps without rebuilding torch-tensorrt or tearing down
# the env — this also sidesteps the test_ext↔quantization `uv sync` conflict,
# which only applies to lockfile resolution. executorch has no group, so it is
# installed as a plain package.

# Install all optional test deps (test-ext + kernels + quantization + executorch)
install-test-ext:
    uv pip install --group test-ext --group kernels --group quantization
    uv pip install pyyaml "executorch>=1.3.1"

# Full L1 tier + test-ext deps, so model-level cases run instead of skipping
tests-l1-ext: install-test-ext tests-l1-dynamo-core tests-l1-dynamo-compile tests-l1-torch-compile tests-l1-torchscript

# Excludes tests-l2-distributed (needs 2+ GPUs and system MPI). Most L2 suites
# are model-level, so run tests-l2-ext (or `just install-test-ext` first) so
# they don't skip.

# Full L2 tier (locally runnable suites)
tests-l2: tests-l2-torch-compile tests-l2-dynamo-compile tests-l2-dynamo-core tests-l2-plugin tests-l2-torchscript

tests-l2-torch-compile *args:
    {{_tier}} trt_tier_l2_torch_compile {{args}}

tests-l2-dynamo-compile *args:
    {{_tier}} trt_tier_l2_dynamo_compile {{args}}

# Also installs executorch (additively) for the executorch/ suite.
tests-l2-dynamo-core *args:
    {{_tier}} trt_tier_l2_dynamo_core {{args}}

# Also installs cuda-python/cuda-core (additively) for the kernels/ QDP suite.
tests-l2-plugin *args:
    {{_tier}} trt_tier_l2_plugin {{args}}

tests-l2-torchscript *args:
    {{_tier}} trt_tier_l2_torchscript {{args}}

# Installs mpich/openmpi via dnf (root-capable container) and runs
# --nproc_per_node=2. Not part of the `tests-l2` aggregate.

# CI-only: needs 2+ GPUs and system MPI
tests-l2-distributed *args:
    {{_tier}} trt_tier_l2_distributed {{args}}

# Full L2 tier + test-ext deps, so model-level cases run instead of skipping
tests-l2-ext: install-test-ext tests-l2-torch-compile tests-l2-dynamo-compile tests-l2-dynamo-core tests-l2-plugin tests-l2-torchscript

# ── Run-all + consolidated failure report ─────────────────────────────────────

# Unlike `just tests-l<N>` (which aborts on the first failing suite), this runs
# every suite and aggregates the JUnit XMLs, so a single consolidated report
# shows all failures — nothing gets clobbered or missed. Append `-ext` to also
# install the test-ext deps first so model-level cases run instead of skipping.
# Pass `--agent` to print the paste-to-Claude Markdown report instead of the
# terminal one — i.e. run + report in one step (e.g. `just tests-report l1-ext
# --agent`). Exits non-zero if anything failed/errored.

# Run a whole tier (l0|l1|l2[-ext]) past failures, then print one report [--agent]
tests-report level *report_args:
    #!/usr/bin/env bash
    set -uo pipefail   # deliberately no -e: keep going past failures
    mkdir -p "$TMPDIR"
    # Accept an optional `-ext` suffix: install the test-ext group first.
    lvl="{{level}}"
    ext=0
    if [[ "$lvl" == *-ext ]]; then ext=1; lvl="${lvl%-ext}"; fi
    case "$lvl" in
      l0) tiers=(l0_converter l0_core l0_py_core l0_torchscript) ;;
      l1) tiers=(l1_dynamo_core l1_dynamo_compile l1_torch_compile l1_torchscript) ;;
      l2) tiers=(l2_torch_compile l2_dynamo_compile l2_dynamo_core l2_plugin l2_torchscript) ;;
      *) echo "unknown level '{{level}}' (use l0|l1|l2, optionally with -ext)" >&2; exit 2 ;;
    esac
    if [[ "$ext" == 1 ]]; then
      # Same set as `just install-test-ext`: pull every optional test-dep group
      # so the model / kernels / quantization / executorch suites run instead of
      # skipping. cuda-core resolves via uv here (the plugin tier's vanilla
      # `pip install cuda-core` cannot fetch it on all hosts).
      echo "==> installing test-ext + kernels + quantization + executorch deps"
      uv pip install --group test-ext --group kernels --group quantization \
        || { echo "test-ext group install failed" >&2; exit 1; }
      uv pip install pyyaml "executorch>=1.3.1" || true
    fi
    results="${RUNNER_TEST_RESULTS_DIR:-$TMPDIR/trt_test_results}"
    rm -f "$results"/*.xml 2>/dev/null || true   # drop stale results from prior runs
    source tests/py/utils/ci_helpers.sh
    export PYTHON="uv run --no-sync python" TRT_JOBS="{{jobs}}" TRT_PYTEST_RERUNS=0
    for t in "${tiers[@]}"; do
      echo "==> trt_tier_$t"
      "trt_tier_$t" || echo "::: trt_tier_$t exited non-zero (continuing) :::"
    done
    # Source of truth is the JUnit XMLs, not exit codes; this sets the final
    # status. Extra args (e.g. --agent) are forwarded to the summary.
    python3 tests/py/utils/junit_summary.py "$results" {{report_args}}

# Reads the JUnit XMLs from the previous run (does not re-run). Pass --agent for
# a plain Markdown report to hand to Claude (`just test-summary --agent`). Exits
# non-zero if that run had failures.

# Print the consolidated failure summary from the last run's JUnit results
test-summary *args:
    python3 tests/py/utils/junit_summary.py {{args}}

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
