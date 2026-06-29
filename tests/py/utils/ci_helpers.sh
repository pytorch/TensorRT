# Shared test-tier library for Torch-TensorRT — the single source of truth for
# "what does each CI tier run". Consumed by BOTH:
#   * CI: .github/workflows/_linux-x86_64-core.yml sources this and calls a
#     trt_tier_* function per job.
#   * Local: the repo-root justfile recipes (just l0, just l1, ...) source this
#     and call the same functions.
#
# Only ENVIRONMENT POLICY differs between the two; the selectors (paths,
# markers, RTX branching) live here once:
#   PYTHON                  pytest/python launcher. CI leaves it unset (-> the
#                           container's ``python``); locally the justfile sets
#                           PYTHON="uv run --no-sync python" so it runs against
#                           the already-built .venv instead of rebuilding.
#   TRT_JOBS                xdist worker count for the parallel suites. CI uses
#                           the default 8; a single local GPU usually wants 2.
#   RUNNER_TEST_RESULTS_DIR where --junitxml files go. Set by CI; locally
#                           defaults to $TMPDIR/trt_test_results.
#   USE_TRT_RTX             "true" selects the TensorRT-RTX test scope (set by
#                           linux-test.yml / the rtx entry workflow).
#
# Each trt_tier_* function forwards any extra args ("$@") to pytest, so you can
# do e.g. `just test`-style narrowing: trt_tier_l0_core -x -k test_foo.

# Absolute repo root, independent of the caller's CWD. This file lives at
# tests/py/utils/ci_helpers.sh, so the repo root is three levels up.
_CI_HELPERS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRT_REPO_ROOT="${TRT_REPO_ROOT:-$(cd "${_CI_HELPERS_DIR}/../../.." && pwd)}"

# python/pytest launcher (override via $PYTHON for local uv runs)
_trt_py() {
    ${PYTHON:-python} "$@"
}

# xdist parallelism token for the parallel suites. TRT_JOBS overrides; otherwise
# falls back to the per-tier default passed as $1 (default 8). L0/L1 use 8 (their
# CI default); L2 uses "auto" (CI relied on the pyproject addopts -n auto). Either
# way TRT_JOBS lets a memory-constrained local GPU throttle every tier.
_trt_nproc() {
    echo "-n ${TRT_JOBS:-${1:-8}}"
}

# --junitxml path for a given suite name (creates the results dir on demand)
_trt_xml() {
    local dir="${RUNNER_TEST_RESULTS_DIR:-${TMPDIR:-/tmp}/trt_test_results}"
    mkdir -p "${dir}"
    echo "${dir}/$1.xml"
}

# trt_pytest wraps pytest with:
#   * --reruns 1 limited to known transient cudagraphs/TRT-driver flakes.
#                 Expand the regex below only with concrete evidence; broad
#                 regexes hide real bugs. Gated on TRT_PYTEST_RERUNS (default
#                 on, for CI) — the local `just` recipes set it to 0 because
#                 the pytest-rerunfailures plugin may not be installed and you
#                 generally want to SEE flakes locally, not silently retry them.
#   * an inline ``::warning::`` reproduce hint on failure.
# Used by the L0/L1 tiers. The L2 tiers call pytest directly (no reruns) to
# match their historical behavior.
trt_pytest() {
    local rerun=""
    if [ "${TRT_PYTEST_RERUNS:-1}" != "0" ]; then
        rerun='--reruns 1 --reruns-delay 5 --only-rerun cudaErrorStreamCaptureInvalidated --only-rerun "Stream capture invalidated"'
    fi
    if ! _trt_py -m pytest $rerun "$@"; then
        # --no-sync so the repro runs against an already-built local checkout
        # instead of uv trying to rebuild torch-tensorrt from source.
        echo "::warning::pytest failed. Reproduce locally with: cd $(pwd) && uv run --no-sync pytest $* (or 'just test ...' from the repo root)"
        return 1
    fi
}

# ── L0: smoke tier ──────────────────────────────────────────────────────────

trt_tier_l0_converter() {
    # Standard TRT shards converter tests with --dist=loadscope; RTX does not.
    local dist=""
    [ "${USE_TRT_RTX:-false}" = "true" ] || dist="--dist=loadscope"
    ( cd "${TRT_REPO_ROOT}/tests/py/dynamo"
      trt_pytest -ra $(_trt_nproc) --junitxml="$(_trt_xml l0_dynamo_converter_tests_results)" ${dist} --maxfail=20 conversion/ "$@" )
}

trt_tier_l0_core() {
    ( cd "${TRT_REPO_ROOT}/tests/py/dynamo"
      trt_pytest -ra $(_trt_nproc) --junitxml="$(_trt_xml l0_dynamo_core_runtime_tests_results)" runtime/test_000_* "$@"
      if [ "${USE_TRT_RTX:-false}" = "true" ]; then
        # RTX runs the whole partitioning suite and skips the hlo subset.
        trt_pytest -ra $(_trt_nproc) --junitxml="$(_trt_xml l0_dynamo_core_partitioning_tests_results)" partitioning/ "$@"
      else
        trt_pytest -ra $(_trt_nproc) --junitxml="$(_trt_xml l0_dynamo_core_partitioning_tests_results)" partitioning/test_000_* "$@"
      fi
      trt_pytest -ra $(_trt_nproc) --junitxml="$(_trt_xml l0_dynamo_core_lowering_tests_results)" lowering/ "$@"
      [ "${USE_TRT_RTX:-false}" = "true" ] || trt_pytest -ra $(_trt_nproc) --junitxml="$(_trt_xml l0_dynamo_hlo_tests_results)" hlo/ "$@" )
}

trt_tier_l0_py_core() {
    ( cd "${TRT_REPO_ROOT}/tests/py/core"
      trt_pytest -ra $(_trt_nproc) --junitxml="$(_trt_xml l0_py_core_tests_results)" . "$@" )
}

# Standard-TRT only (gated off for RTX in the workflow).
trt_tier_l0_torchscript() {
    ( cd "${TRT_REPO_ROOT}/tests/modules" && _trt_py hub.py )
    ( cd "${TRT_REPO_ROOT}/tests/py/ts"
      trt_pytest -ra --junitxml="$(_trt_xml l0_ts_api_tests_results)" api/ "$@" )
}

# ── L1: critical-path tier ────────────────────────────────────────────────────

trt_tier_l1_dynamo_core() {
    ( cd "${TRT_REPO_ROOT}/tests/py/dynamo"
      trt_pytest -ra $(_trt_nproc) --junitxml="$(_trt_xml l1_dynamo_core_tests_results)" runtime/test_001_* "$@"
      [ "${USE_TRT_RTX:-false}" = "true" ] || trt_pytest -ra $(_trt_nproc) --junitxml="$(_trt_xml l1_dynamo_core_partitioning_tests_results)" partitioning/test_001_* "$@"
      trt_pytest -ra $(_trt_nproc) --junitxml="$(_trt_xml l1_dynamo_hlo_tests_results)" hlo/ "$@" )
}

trt_tier_l1_dynamo_compile() {
    ( cd "${TRT_REPO_ROOT}/tests/py/dynamo"
      trt_pytest -m critical -ra --junitxml="$(_trt_xml l1_dynamo_compile_tests_results)" models/ "$@" )
}

trt_tier_l1_torch_compile() {
    ( cd "${TRT_REPO_ROOT}/tests/py/dynamo"
      trt_pytest -ra --junitxml="$(_trt_xml l1_torch_compile_be_tests_results)" backend/ "$@"
      trt_pytest -m critical -ra --junitxml="$(_trt_xml l1_torch_compile_models_tests_results)" --ir torch_compile models/test_models.py "$@"
      trt_pytest -m critical -ra --junitxml="$(_trt_xml l1_torch_compile_dyn_models_tests_results)" --ir torch_compile models/test_dyn_models.py "$@" )
}

# Standard-TRT only.
trt_tier_l1_torchscript() {
    ( cd "${TRT_REPO_ROOT}/tests/modules" && _trt_py hub.py )
    ( cd "${TRT_REPO_ROOT}/tests/py/ts"
      trt_pytest -ra --junitxml="$(_trt_xml l1_ts_models_tests_results)" models/ "$@" )
}

# ── L2: full tier (no reruns; matches historical behavior) ────────────────────

trt_tier_l2_torch_compile() {
    ( cd "${TRT_REPO_ROOT}/tests/py/dynamo"
      _trt_py -m pytest -m "not critical" -ra $(_trt_nproc auto) --junitxml="$(_trt_xml l2_torch_compile_models_tests_results)" --ir torch_compile models/test_models.py "$@"
      _trt_py -m pytest -m "not critical" -ra $(_trt_nproc auto) --junitxml="$(_trt_xml l2_torch_compile_dyn_models_tests_results)" --ir torch_compile models/test_dyn_models.py "$@" )
}

trt_tier_l2_dynamo_compile() {
    ( cd "${TRT_REPO_ROOT}/tests/py/dynamo"
      _trt_py -m pytest -m "not critical" -ra $(_trt_nproc auto) --junitxml="$(_trt_xml l2_dynamo_compile_tests_results)" models/ "$@"
      _trt_py -m pytest -ra $(_trt_nproc auto) --junitxml="$(_trt_xml l2_dynamo_compile_llm_tests_results)" llm/ "$@" )
}

trt_tier_l2_dynamo_core() {
    ( cd "${TRT_REPO_ROOT}/tests/py/dynamo"
      _trt_py -m pytest -ra $(_trt_nproc auto) --junitxml="$(_trt_xml l2_dynamo_core_tests_results)" -k "not test_000_ and not test_001_" runtime/* "$@"
      if [ "${USE_TRT_RTX:-false}" != "true" ]; then
        # ExecuTorch integration is standard-TRT only.
        _trt_py -m pip install pyyaml "executorch>=1.3.1"
        _trt_py -m pytest -ra $(_trt_nproc auto) --junitxml="$(_trt_xml l2_dynamo_executorch_tests_results)" executorch/ "$@"
      fi )
}

trt_tier_l2_plugin() {
    if [ "${USE_TRT_RTX:-false}" = "true" ]; then
        # RTX only runs the automatic-plugin suite (no QDP kernels layer).
        ( cd "${TRT_REPO_ROOT}/tests/py/dynamo"
          _trt_py -m pytest -ra $(_trt_nproc auto) --junitxml="$(_trt_xml l2_dynamo_plugins_tests_results)" automatic_plugin/ "$@" )
    else
        ( cd "${TRT_REPO_ROOT}/tests/py/dynamo"
          _trt_py -m pytest -ra $(_trt_nproc 4) --junitxml="$(_trt_xml dynamo_converters_test_results)" conversion/ "$@"
          _trt_py -m pytest -ra $(_trt_nproc auto) --junitxml="$(_trt_xml dynamo_converters_test_results)" automatic_plugin/test_automatic_plugin.py "$@"
          _trt_py -m pytest -ra $(_trt_nproc auto) --junitxml="$(_trt_xml dynamo_converters_test_results)" automatic_plugin/test_automatic_plugin_with_attrs.py "$@"
          _trt_py -m pytest -ra $(_trt_nproc auto) --junitxml="$(_trt_xml dynamo_converters_test_results)" automatic_plugin/test_flashinfer_rmsnorm.py "$@" )
        # The torch_tensorrt.kernels QDP layer needs cuda-core's high-level
        # ``cuda.core`` API (Device / Program / launch). NVIDIA split this out
        # of the old cuda-python umbrella into the cuda-core distribution for
        # CUDA 13+, so installing cuda-python alone is no longer enough.
        ( _trt_py -m pip install cuda-python cuda-core
          cd "${TRT_REPO_ROOT}/tests/py/kernels"
          _trt_py -m pytest -ra $(_trt_nproc auto) --junitxml="$(_trt_xml dynamo_kernels_test_results)" . "$@" )
    fi
}

# Standard-TRT only.
trt_tier_l2_torchscript() {
    ( cd "${TRT_REPO_ROOT}/tests/modules" && _trt_py hub.py )
    ( cd "${TRT_REPO_ROOT}/tests/py/ts"
      _trt_py -m pytest -ra $(_trt_nproc auto) --junitxml="$(_trt_xml l2_ts_integrations_tests_results)" integrations/ "$@" )
}

# Standard-TRT only; needs a multi-GPU runner + system MPI (CI-only).
trt_tier_l2_distributed() {
    export USE_HOST_DEPS=1
    export CI_BUILD=1
    export USE_TRTLLM_PLUGINS=1
    dnf install -y mpich mpich-devel openmpi openmpi-devel
    ( cd "${TRT_REPO_ROOT}/tests/py/dynamo"
      _trt_py -m pytest -ra -v $(_trt_nproc auto) --junitxml="$(_trt_xml l2_dynamo_distributed_test_results)" \
        distributed/test_nccl_ops.py \
        distributed/test_native_nccl.py \
        distributed/test_export_save_load.py "$@"
      _trt_py -m torch_tensorrt.distributed.run --nproc_per_node=2 distributed/test_native_nccl.py --multirank
      _trt_py -m torch_tensorrt.distributed.run --nproc_per_node=2 distributed/test_export_save_load.py --multirank )
}
