# List all available recipes
default:
    @just --list

# Per-user scratch dir. Torch-TensorRT's engine/timing cache lives under
# $TMPDIR/torch_tensorrt_engine_cache; on shared hosts the default /tmp path is
# created by whoever runs first and then fails for everyone else with a
# PermissionError. Giving each user their own TMPDIR sidesteps that entirely.
export TMPDIR := env_var_or_default("TMPDIR", "/tmp/torch_tensorrt_" + env_var_or_default("USER", "local"))

# pytest xdist parallelism. CI runners use 8; a single local GPU usually can't
# build that many TRT engines at once, so override on smaller machines:
#   just jobs=2 l0
jobs := "auto"

# ── Testing ───────────────────────────────────────────────────────────────────

# Run pytest in the uv-managed env (honors pyproject addopts). Pass any args:
#   just test tests/py/dynamo/conversion/
#   just test -k test_foo -x tests/py/dynamo/runtime/
test *args:
    @mkdir -p "$TMPDIR"
    uv run --no-sync pytest {{args}}

# ── CI tier reproduction ──────────────────────────────────────────────────────
#
# These mirror the pytest selectors in .github/workflows/_linux-x86_64-core.yml
# (standard TensorRT variant) so you can run exactly what a CI tier runs before
# pushing. Keep them in sync with that workflow — it is the source of truth.
# Flags differ on purpose: no --junitxml, no reruns, parallelism via {{jobs}}.

# Full L0 smoke tier
l0: l0-converter l0-core l0-py-core l0-torchscript

l0-converter:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "$TMPDIR"
    cd tests/py/dynamo
    uv run --no-sync pytest -ra -n {{jobs}} --dist=loadscope --maxfail=20 conversion/

l0-core:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "$TMPDIR"
    cd tests/py/dynamo
    uv run --no-sync pytest -ra -n {{jobs}} runtime/test_000_*
    uv run --no-sync pytest -ra -n {{jobs}} partitioning/test_000_*
    uv run --no-sync pytest -ra -n {{jobs}} lowering/
    uv run --no-sync pytest -ra -n {{jobs}} hlo/

l0-py-core:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "$TMPDIR"
    cd tests/py/core
    uv run --no-sync pytest -ra -n {{jobs}} .

l0-torchscript:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "$TMPDIR"
    ( cd tests/modules && uv run --no-sync python hub.py )
    ( cd tests/py/ts && uv run pytest -ra api/ )

# Full L1 tier
l1: l1-dynamo-core l1-dynamo-compile l1-torch-compile l1-torchscript

l1-dynamo-core:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "$TMPDIR"
    cd tests/py/dynamo
    uv run --no-sync pytest -ra -n {{jobs}} runtime/test_001_*
    uv run --no-sync pytest -ra -n {{jobs}} partitioning/test_001_*
    uv run --no-sync pytest -ra -n {{jobs}} hlo/

l1-dynamo-compile:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "$TMPDIR"
    cd tests/py/dynamo
    uv run --no-sync pytest -ra -m critical models/

l1-torch-compile:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "$TMPDIR"
    cd tests/py/dynamo
    uv run --no-sync pytest -ra backend/
    uv run --no-sync pytest -ra -m critical --ir torch_compile models/test_models.py
    uv run --no-sync pytest -ra -m critical --ir torch_compile models/test_dyn_models.py

l1-torchscript:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "$TMPDIR"
    ( cd tests/modules && uv run --no-sync python hub.py )
    ( cd tests/py/ts && uv run pytest -ra models/ )

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
