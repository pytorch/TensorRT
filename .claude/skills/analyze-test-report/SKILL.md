---
name: analyze-test-report
description: "Analyze torch-tensorrt local test results and drive failures to a fix. Use when the user pastes a test report / summary, asks why tests failed, asks to triage or fix failing tests, or mentions the JUnit/test-summary output from `just tests-report` / `just test-summary`. Covers where the JUnit XMLs live, how to read the consolidated report, how to reproduce a single failure, and how to categorize (real bug vs torch-API change vs OOM/skip vs flake)."
---

# Analyzing the torch-tensorrt test report

The local test tiers write **one JUnit XML per pytest suite**, and
`tests/py/utils/junit_summary.py` aggregates them into one report. The JUnit XMLs are
the source of truth — pytest exit codes can be masked when suites run in
sequence, so always reason from the XMLs / the report, not from "the run exited
non-zero".

## Where the output lives

JUnit XMLs are written to (first that is set):
- `$RUNNER_TEST_RESULTS_DIR` — set by CI.
- `$TMPDIR/trt_test_results` — locally. `$TMPDIR` defaults to
  `/tmp/torch_tensorrt_$USER`, so the usual local path is:

  ```
  /tmp/torch_tensorrt_<user>/trt_test_results/*.xml
  ```

Each file is named after its suite, e.g. `l1_dynamo_compile_tests_results.xml`,
`l0_dynamo_core_runtime_tests_results.xml`.

## Getting a report

- **Run a tier and get the agent report in one step** (best for an agent —
  runs every suite past failures, then prints the paste-ready Markdown with node
  ids, file, junit path, repro, message, traceback):
  ```sh
  just tests-report l1 --agent           # l0 | l1 | l2, optionally -ext
  just tests-report l2-ext --agent       # -ext also installs the model-test deps
  ```
  Throttle the GPU with `just jobs=2 tests-report l2 --agent` if it OOMs.
- **Just re-render the last run's report** (no re-run):
  ```sh
  just test-summary --agent              # agent Markdown
  just test-summary                      # color-coded terminal report
  ```
- Or run the script directly on any results dir:
  ```sh
  python3 tests/py/utils/junit_summary.py /tmp/torch_tensorrt_<user>/trt_test_results --agent
  ```

If the user pasted a report, work from it directly. If you need more than it
shows (full traceback), open the `junit:` path it lists.

## Reading the agent report

Each failure block gives you everything to act:
- **`### N. [FAIL|ERROR] classname::name`** — exact pytest node identity.
- **`file:`** — the test source file.
- **`junit:`** — the JUnit XML; read its `<failure>` / `<error>` element for the
  complete traceback (the report caps detail at 40 lines).
- **`repro:`** — a copy-paste command that re-runs the test.
- **`message:` / `detail:`** — the headline and (capped) traceback.

To pull the full traceback for one failure straight from the XML:
```sh
python3 - <<'PY'
import xml.etree.ElementTree as ET
r = ET.parse("<junit-path>").getroot()
for tc in r.iter("testcase"):
    for tag in ("failure", "error"):
        e = tc.find(tag)
        if e is not None:
            print(f"== {tc.get('classname')}::{tc.get('name')} ==")
            print(e.get("message"), "\n", e.text)
PY
```

## Reproducing a failure

Use the `repro` line. Notes that matter on this repo:
- Run via `uv run --no-sync` — uses the already-built `.venv`, does **not**
  rebuild torch-tensorrt. (Plain `uv run` would try to rebuild and fail.)
- `-n0` forces serial (one process). The default pytest config is `-n auto`,
  which spawns a worker per core; on a single GPU that **OOMs** (CUDA out of
  memory + segfaulting workers). For broader local runs use `just jobs=2 ...`.
- Set `TMPDIR=/tmp/torch_tensorrt_<user>` (or just use the `just` recipes, which
  set it) so the TRT engine/timing cache is writable.

Re-run a single test, then the whole suite once it passes:
```sh
TMPDIR=/tmp/torch_tensorrt_$USER uv run --no-sync pytest <file> -k '<name>' -n0
just jobs=2 tests-l1-dynamo-compile          # the suite the failure came from
```

## Categorizing failures (triage before fixing)

- **Real converter/lowering bug** — wrong output, cosine-sim below threshold,
  shape/dtype error in `py/torch_tensorrt/...`. Fix the converter/lowering pass.
- **torch-API change** — `RuntimeError`/`AttributeError` from a torch op whose
  signature/behavior changed in the nightly (the repo tracks torch nightlies).
  Update the call site or the test to the new API; confirm the rule against the
  installed torch before editing (`uv run --no-sync python -c "..."`).
- **OOM / segfault cascade** — `CUDA error: out of memory`, crashed workers.
  Not a code bug: too many xdist workers for the GPU, or the GPU is occupied.
  Re-run with `-n0` / `just jobs=2`; check `nvidia-smi`.
- **Skipped, not failed** — model tests skip without the `test-ext` deps
  (`just install-test-ext`), and RTX/platform-gated tests skip by design.
  Skips are healthy; don't "fix" them.
- **Flake** — passes on re-run with `-n0`. Only the narrow cudagraph stream-
  capture transient is retried in CI (see `tests/py/utils/ci_helpers.sh`).

## Fix loop

1. Get/read the agent report; list the distinct failures and categorize each.
2. For each real failure: read the `junit` traceback, open the `file`, fix.
3. Re-run just that test with its `repro` (serial). Iterate.
4. Re-run the originating suite (`just jobs=2 tests-<tier>`), then
   `just test-summary` to confirm the consolidated report is green.

## Related

- Tier definitions (what each suite runs): `tests/py/utils/ci_helpers.sh`
  (`trt_tier_*`), shared with CI (`.github/workflows/_linux-x86_64-core.yml`).
- Local recipes: `justfile` (`tests-l0/l1/l2[...]`, `tests-report`,
  `test-summary`, `install-test-ext`).
- Building / torch-nightly upgrades: the `build` skill.
