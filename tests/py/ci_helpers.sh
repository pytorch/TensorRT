# Shared shell helpers for Torch-TensorRT CI test scripts.
# Sourced from L0/L1 script blocks in .github/workflows/build-test-linux-x86_64*.yml.
#
# Update this file (not the YAMLs) when adjusting the pytest rerun policy or
# the reproduce-locally hint. Tested only via running CI; if you change a
# function signature, audit every ``source tests/py/ci_helpers.sh`` site.

# trt_pytest wraps ``python -m pytest`` with:
#   * --reruns 1: retry once on known transient cudagraphs/TRT-driver flakes.
#                 Expand the regex below only with concrete evidence; broad
#                 regexes hide real bugs.
#   * an inline ``::warning::`` reproduce hint on failure so reviewers can
#     copy-paste the exact local repro command.
#
# Usage (inside an L0/L1 script: | block):
#     source tests/py/ci_helpers.sh
#     cd tests/py/dynamo
#     trt_pytest -ra -n 8 --junitxml="$RUNNER_TEST_RESULTS_DIR/foo.xml" runtime/test_001_*
trt_pytest() {
    local rerun='--reruns 1 --reruns-delay 5'
    local only_rerun='--only-rerun cudaErrorStreamCaptureInvalidated --only-rerun "Stream capture invalidated"'
    if ! python -m pytest $rerun $only_rerun "$@"; then
        echo "::warning::pytest failed. Reproduce locally with: cd $(pwd) && uv run pytest $*"
        return 1
    fi
}
