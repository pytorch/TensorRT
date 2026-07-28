---
name: build
description: "Build torch-tensorrt locally — install/pin the matching PyTorch nightly, drive Bazel through setup.py, do a clean rebuild after libtorch ABI changes, and recover from common build failures (undefined symbol, stale _C.so, libtorchtrt.so missing). Invoke whenever the user asks to build, rebuild, install editable, or upgrade the torch nightly; or when an import fails with an undefined-symbol error tying torch_tensorrt to libtorch."
---

# torch-tensorrt build skill

## Quick decision tree

1. **C++ ABI undefined-symbol error** (e.g. `undefined symbol: _ZN3c10...c10::ValueError...`) → torch and the compiled `_C.so` / `libtorchtrt.so` are mismatched. Do a **clean rebuild** (delete artifacts + `uv pip install -e .`).
2. **Torch nightly upgrade** (user says "build against MM/DD nightly") → pin torch with `uv pip install` against `https://download.pytorch.org/whl/nightly/cu<MAJOR><MINOR>`, then clean rebuild.
3. **Pure-Python edits** (no `.cpp`/`.h`/`.cc` touched, no torch upgrade) → no rebuild needed. Just re-run.
4. **C++/Bazel edit** → `uv pip install -e . --no-deps --no-build-isolation` (incremental Bazel build) usually suffices. If linkage looks off, escalate to clean rebuild.

## Environment assumptions

- Linux x86_64. Dev venv at `.venv/`. Tooling: `uv`, `bazelisk` (or `bazel`), a matching CUDA toolchain.
- Project pins PyTorch nightly via `pyproject.toml` (the `torch>=…,<…` line in `[project].dependencies`). Bumps occasionally on `main` — check that line if a build fails right after a rebase or branch switch.
- The targeted CUDA version is in `dev_dep_versions.yml` (`__cuda_version__`). Use the matching nightly index, e.g. `https://download.pytorch.org/whl/nightly/cu<MAJOR><MINOR>`.
- TensorRT is fetched by Bazel from `third_party/dist_dir/x86_64-linux-gnu` (no system install required).

## How Bazel finds libtorch

`setup.py:resolve_torch_path()` (see `setup.py:287`) and `MODULE.bazel`/`WORKSPACE`. Detection order:
1. `TORCH_PATH` env var (absolute path to torch package dir)
2. `VIRTUAL_ENV` — used when a venv / `uv` venv is active
3. `CONDA_PREFIX` — when a conda env is active
4. `.venv/bin/python3` relative to repo root
5. `python3` / `python` on `PATH`

If you ever see headers/runtime mismatch, set `TORCH_PATH` explicitly:
```sh
TORCH_PATH=$(uv run python -c "import torch, os; print(os.path.dirname(torch.__file__))") \
    uv pip install -e . --no-deps --no-build-isolation
```

## Commands

### Standard editable install (use this first)
```sh
uv pip install -e . --no-deps --no-build-isolation
```
- Drives Bazel via `setup.py:DevelopCommand.run()` → `build_libtorchtrt_cxx11_abi(develop=True)` → `bazelisk build //:libtorchtrt --compilation_mode=dbg --config=python --config=linux`
- Copies build artifacts via `copy_libtorchtrt()` (untars `bazel-bin/libtorchtrt.tar.gz` into `py/torch_tensorrt/`)
- `--no-deps` avoids re-resolving torch / overwriting a manually-pinned nightly
- `--no-build-isolation` makes Bazel use the active venv's torch (skipping isolation prevents the build from picking up a different torch)
- Build time: ~2 min incremental, ~5 min full

### Clean rebuild (when ABI is mismatched)
```sh
rm -rf build py/torch_tensorrt/_C*.so py/torch_tensorrt/lib/*.so
uv pip install -e . --no-deps --no-build-isolation
```
Required when:
- `import torch_tensorrt` raises `undefined symbol: _ZN3c10...` (libtorch ABI changed)
- After upgrading the torch nightly
- After switching branches that touched C++ code
- After resolving rebase conflicts in `core/runtime/*`

The 4 artifacts to remove:
- `py/torch_tensorrt/_C.cpython-313-x86_64-linux-gnu.so` (Python extension)
- `py/torch_tensorrt/lib/libtorchtrt.so` (main C++ runtime)
- `py/torch_tensorrt/lib/libtorchtrt_runtime.so` (slim C++ runtime)
- `py/torch_tensorrt/lib/libtorchtrt_plugins.so` (TRT plugin shim)
- `build/` (Bazel symlink + scratch)

### Pin a specific torch nightly
The whl index is `https://download.pytorch.org/whl/nightly/cu<MAJOR><MINOR>` — derive `<MAJOR><MINOR>` from `dev_dep_versions.yml` (e.g. CUDA 13.0 → `cu130`). Filename convention: `torch==<X.Y>.<Z>.dev<YYYYMMDD>+cu<MAJOR><MINOR>`, with `<X.Y>` matching the `torch` constraint in `pyproject.toml`.

```sh
# Substitute the right CUDA tag and date; check pyproject.toml for the torch version
uv pip install \
    --index-url https://download.pytorch.org/whl/nightly/cu<MAJOR><MINOR> \
    "torch==<X.Y>.<Z>.dev<YYYYMMDD>+cu<MAJOR><MINOR>"
```
Then **always** clean rebuild afterward — `_C.so` was linked against the previous torch.

### Variants
- **Python-only (no C++ runtime, no TorchScript frontend)**: `PYTHON_ONLY=1 uv pip install -e . --no-deps --no-build-isolation` — skips Bazel entirely. Use when iterating on pure-Python and you don't need engine execution via C++.
- **No TorchScript frontend** (still builds C++ runtime): `NO_TORCHSCRIPT=1 uv pip install -e . --no-deps --no-build-isolation`
- **TRT-RTX build**: `USE_TRT_RTX=1 ...` — rebuilds against the RTX TRT distribution; package name becomes `torch-tensorrt-rtx`.

### Build a wheel (not editable)
```sh
uv pip wheel --no-deps --no-build-isolation -w dist .
```

## Running the build in the background

These builds take 2–5 min. Always launch in the background and poll for terminal markers; do **not** sleep-loop on a fixed schedule:

```python
Bash(
    command="rm -rf build py/torch_tensorrt/_C*.so py/torch_tensorrt/lib/*.so && uv pip install -e . --no-deps --no-build-isolation 2>&1 | tail -5",
    description="Clean rebuild",
    run_in_background=True,
)
# then wait for one of the terminal markers in the background output file:
Bash(
    command="until grep -qE 'Installed|error\\b|ERROR|failed|exit code' \"$OUTPUT_FILE\" 2>/dev/null; do sleep 30; done; tail -10 \"$OUTPUT_FILE\"",
    timeout=600000,
)
```
Or use the `Monitor` tool if available — it notifies on each stdout line and avoids the polling loop entirely.

## Verifying the build worked

```sh
uv run python -c "
import torch, torch_tensorrt
print('torch:', torch.__version__)
print('torch-tensorrt:', torch_tensorrt.__version__)
print('C++ runtime:', torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime)
try:
    print('ABI:', torch.ops.tensorrt.ABI_VERSION())
except Exception:
    pass  # ABI op only registered when the C++ runtime is built
"
```
If this prints clean (no `OSError: Could not load this library: ...libtorchtrt.so`), the build is good.

## Common failure modes

### `undefined symbol: _ZN3c10*` from `libtorchtrt.so` or `_C.so`
torch's libtorch was upgraded but our `.so` files were linked against the old one. **Solution: clean rebuild.**

### `libtorchtrt.so` timestamp from `Jan 1 2000`
That's the wheel-build's reproducible-build epoch. Normal — not stale.

### Bazel says it can't find libtorch
`TORCH_PATH` resolution failed. Verify the venv is active (`echo $VIRTUAL_ENV` should print `.venv`), or set `TORCH_PATH` explicitly (see "How Bazel finds libtorch" above).

### `bazelisk: command not found`
Install bazelisk: `curl -fSsL https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 -o ~/.local/bin/bazelisk && chmod +x ~/.local/bin/bazelisk`. Or install `bazel` matching `.bazelversion`.

### Build succeeds but `import torch_tensorrt` still fails
Old `.so` files may persist if Python cached them. `find py/torch_tensorrt -name __pycache__ -exec rm -rf {} +` then re-import.

### `Permission denied: '/tmp/torch_tensorrt_engine_cache/timing_cache.bin'`
Torch-TensorRT writes its timing cache under `tempfile.gettempdir()` (defaulting to `/tmp`). On shared hosts the cache directory may already exist with another user's ownership, blocking writes. Set `TMPDIR` to a user-private path before running tests/builds:

```sh
export TMPDIR=/tmp/$(whoami)-trt
mkdir -p "$TMPDIR"
```
This makes the engine-cache directory user-scoped and avoids cross-user collisions.

## Key files

- `setup.py` — drives Bazel; respects env vars `PYTHON_ONLY`, `NO_TORCHSCRIPT`, `USE_TRT_RTX`, `RELEASE`, `CI_BUILD`, `TORCH_PATH`, `CU_VERSION`, `JETPACK_BUILD`
- `pyproject.toml` — declares torch version range, build-system `setuptools`, project metadata
- `MODULE.bazel` / `WORKSPACE` — Bazel deps, libtorch detection, TensorRT archive URL
- `.bazelversion` — pinned bazel version (bazelisk reads this)
- `dev_dep_versions.yml` — CUDA / TensorRT version pins surfaced into `__cuda_version__` / `__tensorrt_version__`
- `docsrc/getting_started/installation.rst` — official build docs (Linux, Windows, Jetpack)

## When NOT to rebuild

- Pure Python edits — just re-run.
- `tests/` edits — just re-run pytest.
- `docsrc/` edits — `cd docsrc && make html` (separate flow).
- `.claude/`, `MEMORY.md`, slash-commands — never need a build.
