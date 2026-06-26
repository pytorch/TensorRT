# Recipes source tests/py/ci_helpers.sh (a bash library), so run them in bash.
set shell := ["bash", "-cu"]

# List all available recipes
default:
    @just --list

# Per-user scratch dir. Torch-TensorRT's engine/timing cache lives under
# $TMPDIR/torch_tensorrt_engine_cache; on shared hosts the default /tmp path is
# created by whoever runs first and then fails for everyone else with a
# PermissionError. Giving each user their own TMPDIR sidesteps that entirely.
export TMPDIR := env_var_or_default("TMPDIR", "/tmp/torch_tensorrt_" + env_var_or_default("USER", "local"))

# pytest xdist parallelism for the parallel suites. CI runners use 8; a single
# local GPU usually can't build that many TRT engines at once, so override on
# smaller machines:  just jobs=2 l0
jobs := "auto"

# Launcher + env shared by every tier recipe. The tier definitions themselves
# live in tests/py/ci_helpers.sh — the SAME functions CI calls — so there is a
# single source of truth for what each tier runs. We only set environment
# policy here: PYTHON runs pytest against the already-built .venv (uv --no-sync,
# no rebuild) and TRT_JOBS feeds the parallel suites.
_tier := 'mkdir -p "$TMPDIR" && source tests/py/ci_helpers.sh && export PYTHON="uv run --no-sync python" TRT_JOBS="' + jobs + '" TRT_PYTEST_RERUNS=0 &&'

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
# function from tests/py/ci_helpers.sh — the same one .github/workflows/
# _linux-x86_64-core.yml invokes — so local and CI cannot drift. Extra args are
# forwarded to pytest, e.g. `just l0-core -x -k test_foo`.
# (Standard-TensorRT scope; export USE_TRT_RTX=true before running for RTX.)

# Full L0 smoke tier
l0: l0-converter l0-core l0-py-core l0-torchscript

l0-converter *args:
    {{_tier}} trt_tier_l0_converter {{args}}

l0-core *args:
    {{_tier}} trt_tier_l0_core {{args}}

l0-py-core *args:
    {{_tier}} trt_tier_l0_py_core {{args}}

l0-torchscript *args:
    {{_tier}} trt_tier_l0_torchscript {{args}}

# Full L1 tier
l1: l1-dynamo-core l1-dynamo-compile l1-torch-compile l1-torchscript

l1-dynamo-core *args:
    {{_tier}} trt_tier_l1_dynamo_core {{args}}

l1-dynamo-compile *args:
    {{_tier}} trt_tier_l1_dynamo_compile {{args}}

l1-torch-compile *args:
    {{_tier}} trt_tier_l1_torch_compile {{args}}

l1-torchscript *args:
    {{_tier}} trt_tier_l1_torchscript {{args}}

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
