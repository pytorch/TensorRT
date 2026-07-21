# CI dashboard

A local web UI for Torch-TensorRT's GitHub Actions CI. It answers the questions
the raw Actions tab makes painful: **which platforms are green, which are red,
which test broke, and where that test lives in the tree.**

![what it shows](#) <!-- run it; a screenshot beats this line -->

## Run it

```bash
just ci                       # current branch
just ci branch=nightly        # a specific branch
just ci branch=main port=9000 # pick a port

# or directly (no build — stdlib only):
uv run --no-sync tools/ci-dashboard/ci_dashboard.py -b nightly
python3 tools/ci-dashboard/ci_dashboard.py -b nightly   # also fine
```

It opens `http://127.0.0.1:8712/` locally. It also **binds `0.0.0.0`**, so it's
reachable over your tailnet/LAN — the startup log prints the Tailscale + LAN URLs
(e.g. `http://100.x.y.z:8712/`). To keep it local-only, pass `--host 127.0.0.1`.

### Requirements

- An **authenticated `gh`** — the only data source. Check with `gh auth status`;
  log in with `gh auth login`.
- Python 3.9+ (**stdlib only** — no `pip install`, and **no torch-tensorrt build**).
  `just ci` uses `uv run --no-sync`, which reuses the existing `.venv` interpreter
  and skips the project build; plain `python3` works too since nothing outside the
  stdlib is imported. `htmx` is vendored under `static/`, so it works offline.

## What you get

- **Platform board** — one card per workflow (Linux x86_64, aarch64, Windows,
  the RTX and python-only variants, jetpack…), sorted worst-first, with a live
  status badge. A summary strip up top counts failing / running / queued /
  passing at a glance.
- **Drill in** — open a platform to see its jobs as a **python × cuda × suite**
  grid. Green/red cells; each suite is labelled with the pytest paths it runs
  (read from the `tests/ci` manifest — the same data CI runs).
- **Click a red cell** — a drawer shows every failing test with its error, the
  **source file:line it maps to** (local path + a GitHub link pinned to the run's
  commit), and a **copy-paste command to reproduce it locally** — the exact
  `python -m tests.ci run <suite>` command CI used, narrowed to the failing tests.
- **Relevant logs, captured** — the drawer pulls that job's log (ANSI codes and
  `##[group]` markers stripped) and shows just the part that matters:
  - the **pytest FAILURES block per failing test** (traceback + captured output),
    so you read the actual failure without scrolling a ~0.5 MB / ~8k-line log;
  - **root-cause collapse** — when N failures share one reason (e.g. 5 cumsum tests
    with `build_serialized_network returned None`, or 29 tests that all failed to
    import), they fold into **one** `N× …` block + a banner, not N identical ones;
  - **install/build failures surfaced** — an import storm (every test fails to
    import) points at the actual `pip … subprocess-exited-with-error`, not the
    misleading per-test `ModuleNotFoundError`s;
  - **smart anchoring** for non-pytest failures — instead of the blind log tail, it
    anchors on the real error (`short test summary`, a build error, or a backward
    walk from `exit code N` to the traceback), skipping echoed `::error::` script noise;
  - **search / jump / highlight** — a search box (with a *matches only* filter), an
    error **jump-list** that scrolls to each error line, and severity-coloured lines.
  A **raw log ↗** link still opens the full plain-text log to grep/save.
- **Failures across platforms** — a rollup that dedupes a failing test across
  every platform it breaks on ("fails on 3 platforms → likely this code"), so a
  systemic break stands out from a one-off.
- **"Didn't run" ≠ "failed"** — skipped/gated-off tiers (common on PR branches),
  jobs blocked by a failed dependency, and cancelled/never-started jobs are shown
  as a distinct, dashed/muted **didn't run** state — never colored red and never
  counted as failing. The summary strip tallies them separately, and the
  cross-platform rollup only ever counts jobs that actually *ran and failed*.
- **Live** — status badges refresh in the background (~every 25s) without
  collapsing whatever you've expanded. `↻ Refresh` (or `r`) forces a full reload;
  `/` focuses the branch box; the *failing only* toggle hides the green cards.

## PR commands

Comment these on a PR to control its CI without pushing a new commit (write
access required; handled by `.github/workflows/retrigger-ci.yml`). They're also
listed in the dashboard header under **PR commands** with copy buttons.

| comment | effect |
|---|---|
| `/rerun` | re-run only the failed / cancelled jobs on the current commit |
| `/rerun all` | re-run every job from scratch on the current commit |
| `/cancel` | cancel stale “zombie” runs still in-flight on **old** commits of the PR |
| `/cancel all` | cancel **every** in-flight run for the PR (full stop) |
| `/test <lane> [backend]` | dispatch a fresh run at an exact lane — `lane` ∈ `fast\|full\|nightly`, `backend` ∈ `standard\|rtx\|both` (default `both`). e.g. `/test full rtx`, `/test nightly` |

Typical unstick: `/cancel` to clear zombies, then `/rerun all` for a clean wave.
Run everything on demand: `/test full` (or `/test nightly` for llm/kernels/distributed).
(These take effect from `main`, since `issue_comment` workflows always run there.)

## How it works

The server (`ci_dashboard.py`) is a thin, caching proxy over `gh`; all rendering
is server-side HTML fragments and `htmx` wires up the interactions (lazy-load a
platform on open, out-of-band badge polling, the failure drawer).

- **Runs / jobs**: `gh run list` + `gh api …/actions/runs/{id}/jobs`. Job names
  encode the matrix — `core / L2 dynamo distributed tests /
  L2-dynamo-distributed-tests--3.12-cu130` — which we parse into
  `{tier, python, cuda, kind}`.
- **Why a red cell → a test → a file**: the test jobs surface each pytest
  failure as a GitHub **check-run annotation** (via `pytest-results-action`), so
  the failing test name + traceback is one cheap API call
  (`…/check-runs/{id}/annotations`) — no wheel or junit download. We then
  `git grep` the test symbol in `tests/` to resolve it to `file:line`.
- **Suite → paths / reproduce command** come straight from the `tests/ci`
  manifest (`tests/ci/suites.py`) via `info_for()` — the CI names each test job
  `<suite>-<variant>`, which is exactly a manifest suite, so there is nothing to
  keep in sync. A small `TIER_MAP` remains only as a fallback so historical,
  pre-migration runs (tier-named) still resolve; it can be dropped once those age
  out.

## Limitations

- Results are only as granular as the CI annotations. A **build/env/setup**
  failure (not a test assertion) has no pytest annotation — the drawer says so,
  then anchors the log on the real error (and, for an import storm, points at the
  install failure) and links you to the raw job log.
- Expanded platform grids don't auto-refresh (only the top-level badges do). Open
  a stale platform again, or hit `↻ Refresh`, to re-pull its jobs.
- The cross-platform rollup fans out `gh` calls for every failing test job; on a
  branch that's failing everywhere the first scan takes a few seconds (results
  are cached; `↻ rescan` forces a refresh).
