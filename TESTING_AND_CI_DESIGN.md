# Torch-TensorRT — Testing & CI Design

> **Status:** proposal / north-star design. Some pieces already exist on `main`
> (marked **✓ exists**); the rest is the target end-state and a rollout plan.
>
> **One-line goal:** *a developer can run exactly what CI runs, locally, with one
> command — and when something fails, the failure tells them how to reproduce and
> fix it.*

---

## 1. Goals & non-goals

**Goals**
- **Full coverage** of torch-tensorrt: converters, runtime, lowering, dynamo, torch-compile, torchscript, plugins, models, distributed — across the platforms/CUDA/Python we ship.
- **Ergonomic for developers**: the local experience is the *primary* design target, not an afterthought bolted onto CI.
- **Zero local/CI drift**: the command that runs a tier in CI is the *same* command you run locally.
- **Fast feedback by default, full coverage on demand**: a PR gets a signal in ~15 min; the exhaustive matrix runs when it matters.
- **Failures are never hidden**: nothing gets clobbered, every failure prints its own reproduce command, and the aggregate is machine- and agent-readable.

**Non-goals**
- Replacing the PyTorch test-infra build plumbing (we reuse it for wheels/runners).
- A bespoke remote-execution cluster (we explicitly assume **no owned cache/RBE infra**).
- 100% of the matrix on every push (that's what nightly + the full lane are for).

---

## 2. Design principles — the "pleasant" contract

These are the invariants. Every decision below serves one of them.

1. **One command to run anything.** `just <thing>` — never a 200-char `pytest` incantation.
2. **Local == CI.** Suites are a declarative manifest + one runner; `just` and CI both call `ci run <suite>`, so there's nothing to drift.
3. **No rebuild to test.** Running tests never triggers a Bazel rebuild.
4. **Reproduce in one line.** Every CI failure prints `uv run --no-sync pytest … -n0 <node>`.
5. **Build once, test many.** One wheel per (platform, CUDA, Python); every tier reuses it.
6. **Fail loud, aggregate everything.** One consolidated, agent-friendly report; no silent truncation; no fragile third-party glue.
7. **Cheap to retrigger, easy to discover.** Re-run via a PR comment; a menu tells you how.
8. **Pay for the signal you need.** Tiers + lanes + path filters; don't run GPU jobs for a docs typo.
9. **Flakes never mask real failures.** Retries are scoped to known signatures; everything else is quarantined with a tracking issue.

---

## 3. Mental model

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 0 — Test logic (platform-agnostic, the single source)      │
│    tests/ci/suites.py        SUITES = [Suite(...)]   manifest (DATA)│
│    tests/ci/__main__.py      `ci run <s>` · `ci matrix`  one engine │
│    @pytest.mark.{smoke,critical}    lane membership on the test     │
│    tests/py/utils/junit_summary.py  aggregate + --agent report  ✓  │
│    tests/py/utils/ci_helpers.sh     env + trt_pytest wrapper only  │
│    pyproject.toml                   markers, optional dep groups   │
└───────────────┬───────────────────────────────┬───────────────────┘
                │ calls "ci run <suite>"          │ matrix from "ci matrix"
        ┌───────▼────────┐                ┌───────▼─────────────────┐
        │ LAYER 1 — LOCAL │                │ LAYER 1 — CI            │
        │   justfile  ✓   │                │   _build.yml / _test.yml│
        │   uv, jj fix    │                │   ci.yml (lanes)        │
        └────────────────┘                └───────┬─────────────────┘
                                                   │
                                          ┌────────▼─────────┐
                                          │ LAYER 2 — STATUS │
                                          │  rollup check    │
                                          │  PR comment(--agent)│
                                          │  /rerun, menu  ✓  │
                                          └──────────────────┘
```

The point: **Layer 0 is the contract — and it is _data + one engine_, not a pile of
shell functions.** A declarative **suite manifest** (`tests/ci/suites.py`) says *what*
each suite is; a single **runner** (`tests/ci/__main__.py`) is the *only* place that knows
*how* to turn a suite into a `pytest` command. `just` and CI are both thin callers of
`ci run <suite>`, and CI's job matrix is *generated* by `ci matrix` — so they cannot drift,
and adding a job is a one-line data change (§5.3).

---

## 4. Test taxonomy

### 4.1 Two orthogonal axes (retiring L0/L1/L2-by-filename)

A job is a cell in **(subsystem × lane × variant)** — three *independent* axes. The old
`L0/L1/L2` tier numbers fused "what subsystem" with "how deep," and worse, encoded depth
in **filenames** (`runtime/test_000_*` = L0, `runtime/test_001_*` = L1, the rest = L2 via
`-k "not test_000_…"`). That's retired. The axes are now:

- **Subsystem** *(what — a directory)*: `converters`, `runtime`, `lowering`, `partitioning`, `dynamo-models`, `torch-compile`, `torchscript`, `plugins`, `kernels`, `quantization`, `llm`, `distributed`, `executorch`.
- **Lane** *(how deep — a marker)*: `fast` (= `-m smoke`, every push), `full` (default, the ready-signal lane), `nightly` (everything + perf).
- **Variant** *(where — a dimension)*: `standard` / `rtx` / platform — applied centrally by the runner, **not** as `if [ "$USE_TRT_RTX" ]` branches scattered through shell.

|              | fast (every push) | full (ready signal) | nightly |
|---|---|---|---|
| converters | smoke subset | all | + fuzz |
| runtime / lowering / partitioning | smoke subset | all | — |
| dynamo-models / torch-compile | — | `-m critical` | all + llm |
| torchscript | api smoke | models | integrations |
| plugins / kernels / quantization | — | — | all (+optional deps) |
| distributed | — | — | multi-GPU |

Depth lives on the test (a marker), so **moving a test between lanes is editing a
decorator, never renaming a file.** Sharding a big subsystem is a manifest field
(`shards: N`), not a `test_000_*` filename convention.

### 4.2 Lane membership = markers, uniformly (`pyproject.toml`)
Membership is expressed *one* way — a marker on the test — never filename prefixes or
`-k "not test_000_…"` strings (today three different mechanisms coexist for the same idea).
- `smoke` *(proposed)* — the fast-lane subset of any subsystem.
- `critical` ✓ — the `full`-lane core; nightly runs the complement (`-m "not critical"`). Clean partition, no overlap, no gap.
- `unit` ✓ — the bulk of the suite.
- `flaky` *(proposed)* — quarantined out of gating lanes, tracked by an issue, still run nightly for visibility.

### 4.3 Optional dependency groups (`-ext`)
`test-ext`, `kernels`, `quantization` (uv groups) + `executorch` (plain package). Suites that need them **skip cleanly when absent** (e.g. `conftest.py` `skip_no_cuda_core`), so a base checkout never hard-fails. `just install-test-ext` pulls them all.

### 4.4 Directory layout (`tests/py/`)
```
tests/
  ci/                      # ← NEW: the Layer-0 brain — suites.py (manifest) + __main__.py (runner)
  py/
    dynamo/{conversion,lowering,runtime,partitioning,models,hlo,llm,executorch,…}
    ts/                    # torchscript frontend
    kernels/  quantization/   # gated on optional deps
    distributed/           # multi-GPU
    utils/                 # ci_helpers.sh (env + trt_pytest wrapper), junit_summary.py
```

---

## 5. Layer 0 — the suite manifest + one runner

This replaces the bash `trt_tier_*` library as the source of truth. The bash version
*works*, but config-in-shell conflates data with mechanism and quietly hides errors. Today
it: tiers tests by **filename prefix**; mixes markers / `-k` strings / globs for the same
"depth" concept; drops the L2 suites out of the `trt_pytest` rerun+repro wrapper (so L2
failures get no repro hint); and — really — points **four** pytest runs at one
`--junitxml` path so three suites' results vanish from the aggregate. Those are all
symptoms of the same root cause, fixed by separating data from engine.

### 5.1 The manifest — `tests/ci/suites.py` *(data; typed, mypy-validated)*
```python
@dataclass(frozen=True)
class Suite:
    name: str                              # "dynamo-runtime"
    paths: tuple[str, ...]                 # ("dynamo/runtime",) under tests/py
    lanes: tuple[Lane, ...] = ("full",)    # fast | full | nightly
    shards: int = 1                        # split a big suite across N CI jobs
    dist: str | None = None                # "--dist=loadscope"
    needs: tuple[str, ...] = ()            # optional dep groups: ("kernels", "cuda-core")
    variants: tuple[Variant, ...] = ("standard", "rtx")
    extra_args: tuple[str, ...] = ()       # ("--ir", "torch_compile"), ("--maxfail", "20")

SUITES = [
    Suite("dynamo-converters", ("dynamo/conversion",), lanes=("fast", "full"),
          dist="--dist=loadscope", shards=4),
    Suite("dynamo-runtime",    ("dynamo/runtime",),    lanes=("fast", "full")),
    Suite("dynamo-models",     ("dynamo/models",),     lanes=("full", "nightly")),
    Suite("kernels",           ("kernels",),           lanes=("nightly",),
          needs=("kernels", "cuda-core"), variants=("standard",)),
    # … one line per subsystem
]
```
A typo'd field or path is a **static error**, not a silently-empty glob. (YAML + a pydantic
schema is the friendlier-for-non-coders alternative; we pick Python for the static checking
and because the same module generates the CI matrix.)

### 5.2 The runner — `tests/ci/__main__.py` *(the ONLY place pytest mechanics live)*
```bash
python -m tests.ci matrix --lane fast --variant standard   # → JSON for the GH Actions matrix
python -m tests.ci run dynamo-runtime --shard 2/4          # builds + runs pytest
```
The runner — and nothing else — knows how to: apply the lane's marker (`fast` → `-m smoke`),
derive a *unique* junit path from `name+shard` (no more collisions), wrap **every** suite in
the `trt_pytest` reruns+repro helper *uniformly*, apply `--dist`/`extra_args`, gate on `needs`
(skip cleanly if a dep group is absent), and resolve variants centrally. Its knobs:

| Knob | Default | Meaning |
|---|---|---|
| `PYTHON` | `python` | CI: container python; local: `uv run --no-sync python` (no rebuild) |
| `TRT_JOBS` | per-suite | xdist `-n`; **GPU-memory-aware, not `-n auto`** (one GPU OOMs long before it runs out of cores) |
| `TRT_PYTEST_RERUNS` | `1` (CI) / `0` (local) | gated reruns for known flake signatures only |
| `RUNNER_TEST_RESULTS_DIR` | `$TMPDIR/trt_test_results` | where JUnit XMLs land |
| `TMPDIR` | per-user | engine/timing cache; per-user to dodge cross-user permission collisions |

> **Lesson baked in:** rerun args are a Python list passed to `subprocess` (or, in the
> shrunken shell wrapper, a bash *array*) — **never** an unquoted string. A multi-word
> `--only-rerun "Stream capture invalidated"` expanded unquoted word-splits into phantom
> test paths and silently collects **0 tests** — this is exactly how a real CI run failed.

### 5.3 Adding a job — one line
- **Before** (bash): write a `trt_tier_*` function in `ci_helpers.sh` + a `just` recipe + a matrix entry in `_test.yml` + a `tests-report` tier-list entry — *4 edits, 3 files, in bash.*
- **After**: add one `Suite(...)` to `SUITES`. `just` exposes it, CI's matrix includes it (it's generated), the report aggregates it — **automatically.**

### 5.4 `tests/py/utils/ci_helpers.sh` — shrunk, not deleted ✓exists
Keeps only the genuinely-shell bits the runner shells out to: env setup and the
array-safe `trt_pytest` wrapper. No tier definitions, no selection logic.

### 5.5 `tests/py/utils/junit_summary.py` ✓exists
Reads all JUnit XMLs and emits a **human** report (TTY colors, `NO_COLOR`/`FORCE_COLOR`) and
an **`--agent`** report: Markdown with node id, file, junit path, a copy-paste
`uv run --no-sync pytest <file> -k <name> -n0` repro, message, and capped detail — *paste it
to Claude and it can start fixing.* Exits non-zero on any failure **or empty result set**
(the XMLs are the source of truth for pass/fail). No third-party result actions that crash
on empty input.

---

## 6. Local developer experience

### 6.1 The golden path
```bash
just test tests/py/dynamo/conversion/test_foo.py   # inner loop: one file, fast
just tests-l0                                       # the smoke tier, exactly as CI runs it
just tests-report l1-ext --agent                    # run a whole tier past failures → paste-ready report
just lint                                           # all pre-commit hooks (== the linter CI job)
```
Everything is `uv run --no-sync` underneath — **tests never rebuild torch-tensorrt.**

### 6.2 Recipes (justfile — a thin caller of the runner)
- `test *args` — raw pytest in the uv env (honors `pyproject` addopts).
- `suite <name> [-- pytest args]` — run one manifest suite, *exactly* as CI runs it (`ci run <name>`).
- `lane <fast|full|nightly>` — run every suite in a lane locally.
- `report <lane> [--agent]` — **run a lane past failures, then print one consolidated report** (run + report in one step).
- `summary [--agent]` — re-print the last run's report without re-running.
- `lint` / `lint-changed` — pre-commit over all / changed files.
- `build` *(proposed)* — wrap the clean-rebuild flow; auto-detect ABI staleness after a nightly bump.
- `jobs=N` knob — raise xdist parallelism when your GPU has headroom (`just jobs=8 lane full`).

During migration the legacy `tests-l0/l1/l2` recipes become thin aliases for the matching lanes, then retire.

### 6.3 Build ergonomics
- `uv pip install -e . --no-deps --no-build-isolation` for incremental; **clean rebuild after a torch-nightly bump** (libtorch ABI changes → `undefined symbol` otherwise).
- `PYTHON_ONLY=1` for pure-Python iteration (skips Bazel entirely).
- A `build` skill / recipe encodes the decision tree so nobody re-learns it.

### 6.4 Formatting is automatic, not a chore
- `pre-commit` for the full gate (the checks: mypy, validate-pyproject, uv-lock, …).
- **`jj fix`** wired to the *formatters* (black, ruff, isort, clang-format, buildifier) as stdin→stdout filters, version-pinned via `uvx` to the exact pre-commit revs → format an entire stack in one shot, no per-commit `pre-commit` dance.

### 6.5 The failure→fix loop
Any failing tier → `just test-summary --agent` → the `analyze-test-report` skill triages (real bug vs torch-API change vs OOM/skip vs flake) and drives each to a fix using the printed repro commands.

---

## 7. CI design

### 7.1 Lanes — *without a merge queue*

We **cannot** use GitHub's merge queue, so the full suite is gated by an explicit
"ready" signal instead of a queue:

| Lane | Trigger | What runs | Required? |
|---|---|---|---|
| **Fast** | every PR push | lint + **1 representative build** (py-latest + newest CUDA, standard) + **L0** | informational |
| **Full** | `ci: full` label · `/ci full` comment · **approval** (`pull_request_review`) | full matrix · L1/L2 · RTX · all platforms | **yes** (`CI / full` rollup) |
| **Main canary** | `push` to `main` | full lane | — (catches trunk breakage → fast revert) |
| **Nightly** | `schedule` | full + `-ext` model/kernels/quant + perf + exhaustive CUDA/Python | — |

**Gating mechanics.** Branch protection requires `CI / full`. It only reports after
the full lane runs (on label/approval). Pair with **"dismiss stale approvals on new
commits"** so a post-approval push invalidates approval → re-approve → full re-runs on
the new HEAD.

> **Honest gap.** Without a merge queue we lose the guarantee that the *actually-merged*
> tree (after other PRs land) was tested as a unit — two independently-green PRs can break
> `main` together. Mitigation: (a) require PRs rebased on fresh `main` before the full run,
> (b) the main canary + fast revert. This is the inherent cost of no queue; we accept it.

### 7.2 Topology — build once, test many ✓(x86_64 today)

```
ci.yml  (one entry: pull_request | pull_request_review | push | schedule | comment)
  ├─ _build.yml   matrix{platform, cuda, python, variant} → ONE wheel artifact each
  └─ _test.yml    download wheel → run tier(s) via ci_helpers.sh → JUnit artifact
        └─ rollup job   aggregate JUnit → single status/platform + PR comment
```

Platform / RTX / python-only stop being **separate workflow files** and become
**matrix inputs**. This collapses today's ~11 near-duplicate entry workflows (incl. the
461-line inline Windows files) into `_build.yml` + `_test.yml` + a thin `ci.yml`.

### 7.3 Caching — *without owned infra*

We assume **no remote cache/RBE server**. Use GitHub's own cache, two layers, both wired
into the **vendored** build workflow (it's a local copy, so we can edit its steps):

1. **sccache with the GitHub Actions backend** (`SCCACHE_GHA_ENABLED=true`) — free, no infra; caches C++ object compiles. Biggest win for Bazel C++ recompiles.
2. **Bazel `--disk_cache=<dir>` persisted via `actions/cache@v4`** — key on hashes of `MODULE.bazel` / `.bazelversion` / **torch-nightly + CUDA version** + `restore-keys` for partial hits.

**Container caveat:** the build runs inside a manylinux container; `actions/cache` runs on
the host. Either **bind-mount** a host dir as the Bazel disk-cache path (no tokens cross
the boundary — preferred), or forward `ACTIONS_CACHE_URL`/`ACTIONS_RUNTIME_TOKEN` into the
container for sccache.

**Warming model (forks for free):** populate caches on `push` to `main` (trusted, can
write); PRs — **including forks** — restore **read-only**. Forks get no secrets and can't
write the base cache, but they *can* read main's warm cache, so a fork PR still builds
incrementally with zero secret/infra exposure.

> Caveats: GHA cache is **10 GB/repo, LRU-evicted** — scope it (sccache's compact object
> cache fits better than a raw disk_cache). **Key on the nightly/CUDA version** so a bump
> busts it — a stale cache here is exactly the ABI-mismatch class of bug. Step-up option if
> limits bite: BuildBuddy's free *hosted* tier (just an API key; non-fork PRs + main).

### 7.4 Status & reporting
- **One rollup check per platform** (`CI / Linux x86_64`, …) summarizes its child jobs → branch protection keys on a *stable* name; reviewers see one green/red + a Markdown table. ✓exists
- JUnit from every job is aggregated by `junit_summary.py` into **one artifact + an auto PR comment** (the `--agent` report). Nothing clobbered, nothing missed.
- `--fail-on-empty` so "0 tests collected" is a failure, not a silent green.

### 7.5 Ergonomic CI commands
- **`/rerun`** / **`/rerun all`** — re-run failed/cancelled or everything, no new commit (write-access gated). ✓exists
- **PR command menu** — on PR open, a comment lists the commands + local-repro recipes (so contributors discover them). ✓built
- **`/ci full`** *(proposed)* — opt a PR into the full lane (the merge-queue substitute trigger).

### 7.6 Flake handling
- **Gated reruns for known signatures only** (`--only-rerun cudaErrorStreamCaptureInvalidated`, "Stream capture invalidated"). Never a blanket retry — that masks real failures.
- **Quarantine** `@pytest.mark.flaky` out of gating lanes with a tracking issue; still run nightly for visibility.
- **Cache model weights** as an artifact → kills the corrupted-download flakes (e.g. the mobilenet `unexpected EOF` torchscript failure).

### 7.7 Concurrency & path filters
- `concurrency: cancel-in-progress` keyed on PR ref — never burn runners on superseded commits.
- **Path filters**: docs-only PR → skip GPU/C++; `.py`-only → skip the C++ test tier. (The single biggest waste-cutter on trivial PRs.)
- Skip CI on draft PRs; fast lane on "ready for review".

---

## 8. Target file layout

```
.github/workflows/
  ci.yml              # the ONLY build+test entry: lanes by event
  _build.yml          # reusable: matrix → wheel artifact (all platforms/variants)
  _test.yml           # reusable: wheel → suite(s) → JUnit; matrix GENERATED by `ci matrix`
  nightly.yml         # exhaustive matrix + -ext + perf
  retrigger-ci.yml    # /rerun                          ✓exists
  pr-command-menu.yml # discoverability comment on open  ✓built
  linter.yml          # pre-commit gate
  release-*.yml       # publish on tags (unchanged)
tests/ci/
  suites.py           # ← the manifest (DATA): SUITES = [Suite(...)]
  __main__.py         # ← the runner/engine: `ci run` · `ci matrix`
tests/py/utils/
  ci_helpers.sh       # SHRUNK: env + array-safe trt_pytest wrapper only ✓exists
  junit_summary.py    # aggregation + --agent report    ✓exists
justfile              # local entry — thin caller of `ci run`/`ci matrix` ✓exists
pyproject.toml        # markers + optional dep groups    ✓exists
TESTING_AND_CI_DESIGN.md   # this document
```

**Deletions this enables:** the ~11 per-platform entry workflows, the parallel
schedule/dispatch build-test set (the source of the old/new *double-run*), and the
inline 461/381-line Windows workflows — all collapse into `_build.yml` + `_test.yml`.

---

## 9. Rollout plan (incremental, each step shippable + verified)

Use a **byte-equivalence harness** at each step — diff `ci run <suite>`'s emitted `pytest`
command against the bash tier it replaces, and render the workflow's generated script —
before flipping it on. Prove we didn't change *what runs*, only *how it's wired*.

0. **Kill the double-run.** Delete/disable the superseded schedule/dispatch build-test set so a PR runs *one* set of jobs. (Halves PR cost, removes the confusion that surfaced in #4352.)
1. **Stand up the manifest + runner.** Port the `trt_tier_*` functions into `tests/ci/suites.py` + `tests/ci/__main__.py` one suite at a time, each producing a byte-identical command to the bash tier it replaces. Add `smoke` markers to replace the `test_000_*`/`test_001_*` filename tiering; shrink `ci_helpers.sh` to env + wrapper.
2. **Generalize the core** → `_build.yml` + `_test.yml` with a `platform`/`variant` input; `_test.yml`'s matrix is generated by `ci matrix`. Fold RTX + python-only in as variants.
3. **Migrate aarch64 + Windows** onto the same reusables (deletes the inline Windows files).
4. **Lanes** — add `ci.yml` orchestrator + the fast/full split + approval/label/`/ci full` triggers + `concurrency`.
5. **Nightly** — `nightly.yml` for the exhaustive matrix + `-ext`; trim the PR lane to fast.
6. **Caching** — sccache + Bazel disk_cache via GHA cache, warmed on main.
7. **Polish** — path filters, weight-cache, flake quarantine, drop fragile result actions.

---

## 10. Open decisions

- **Runner availability** for GPU jobs (how many parallel full lanes can we afford?).
- **GHA cache budget** — is 10 GB enough, or do we want BuildBuddy's free hosted tier?
- **Required-check names** for branch protection (must stay stable across the migration).
- **Fork policy** — confirm fork PRs are read-only on cache and never see secrets.
- **`/ci full` vs approval-only** as the full-lane trigger (or both).

---

## 11. What "pleasant" feels like, end-to-end

> Open a PR → a comment greets you with the command menu → a **fast green check in ~15 min**.
> Iterate on the fast lane. When ready, a reviewer approves (or you comment `/ci full`) → the
> full suite runs once. A failure? **One PR comment** with the exact `uv run --no-sync pytest …`
> for each failing test — paste it to Claude, or run it locally via `just`. Flaky? `/rerun`.
> Formatting? `jj fix`. You never assembled a `pytest` command by hand, never waited on a job
> irrelevant to your change, and never had a real failure hidden behind a flaky one.
