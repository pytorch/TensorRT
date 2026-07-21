#!/usr/bin/env python3
"""Torch-TensorRT CI dashboard — a local, pleasant view of GitHub Actions.

Run:  uv run tools/ci-dashboard/ci_dashboard.py            # current branch
      uv run tools/ci-dashboard/ci_dashboard.py -b nightly # a branch
      python3 tools/ci-dashboard/ci_dashboard.py --port 8712

No pip dependencies. Data comes from the `gh` CLI (must be authenticated:
`gh auth status`). The server renders HTML fragments; htmx (vendored under
./static) drives lazy-loading, live status polling, and the failure drawer.

Why annotations, not artifacts: the test jobs surface every pytest failure as a
GitHub check-run *annotation* (via pytest-results-action), so the failing test
name + traceback is one cheap API call away — no wheel/junit download needed.

Job names encode the whole matrix, e.g.
    test / dynamo-converters-standard / dynamo-converters-standard--3.12-cu130
We parse that into {group, python, cuda, kind}; the group is a `<suite>-<variant>`
from the tests/ci manifest (tests/ci/suites.py — the SAME data CI runs), so a red
cell maps to the suite's pytest paths and its exact `python -m tests.ci run`
reproduce command. Pre-migration runs (tier-named) fall back to the legacy map.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# ── tests/ci manifest (single source of truth) ───────────────────────────────
# The dashboard resolves a red CI cell back to "what ran + how to reproduce" from
# the SAME manifest CI and the `just` recipes use (tests/ci/suites.py). CI names
# each test job `<suite>-<variant>` (tests/ci → linux-test.yml), so the group the
# dashboard parses out of a job IS a manifest suite — no duplicated path lists to
# drift. Imported best-effort: a checkout predating the manifest falls back to the
# legacy TIER_MAP below.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
try:
    from tests.ci import SUITES as _SUITES  # noqa: E402
except Exception:  # pragma: no cover — manifest absent on pre-migration branches
    _SUITES = ()
try:
    from tests.ci.runner import matrix as _ci_matrix  # noqa: E402
except Exception:  # pragma: no cover
    _ci_matrix = None

# ── Legacy tier → source mapping (fallback) ──────────────────────────────────
# Kept only so historical, pre-migration runs (named "L2 dynamo core tests", not
# "<suite>-<variant>") still resolve. New runs go through the manifest above.
# Keys are normalized job group names (see _norm()); `fn` is the old shell tier.
TIER_MAP = {
    "l0 dynamo converter tests": dict(
        fn="trt_tier_l0_converter", paths=["tests/py/dynamo/conversion/"]
    ),
    "l0 dynamo core tests": dict(
        fn="trt_tier_l0_core",
        paths=[
            "tests/py/dynamo/runtime/test_000_*",
            "tests/py/dynamo/partitioning/test_000_*",
            "tests/py/dynamo/lowering/",
            "tests/py/dynamo/hlo/",
        ],
    ),
    "l0 core python tests": dict(fn="trt_tier_l0_py_core", paths=["tests/py/core/"]),
    "l0 torchscript tests": dict(
        fn="trt_tier_l0_torchscript", paths=["tests/py/ts/api/", "tests/modules/hub.py"]
    ),
    "l1 dynamo core tests": dict(
        fn="trt_tier_l1_dynamo_core",
        paths=[
            "tests/py/dynamo/runtime/test_001_*",
            "tests/py/dynamo/partitioning/test_001_*",
            "tests/py/dynamo/hlo/",
        ],
    ),
    "l1 dynamo compile tests": dict(
        fn="trt_tier_l1_dynamo_compile", paths=["tests/py/dynamo/models/"]
    ),
    "l1 torch compile tests": dict(
        fn="trt_tier_l1_torch_compile",
        paths=[
            "tests/py/dynamo/backend/",
            "tests/py/dynamo/models/test_models.py",
            "tests/py/dynamo/models/test_dyn_models.py",
        ],
    ),
    "l1 torch script tests": dict(
        fn="trt_tier_l1_torchscript", paths=["tests/py/ts/models/"]
    ),
    "l2 torch compile tests": dict(
        fn="trt_tier_l2_torch_compile",
        paths=[
            "tests/py/dynamo/models/test_models.py",
            "tests/py/dynamo/models/test_dyn_models.py",
        ],
    ),
    "l2 dynamo compile tests": dict(
        fn="trt_tier_l2_dynamo_compile",
        paths=["tests/py/dynamo/models/", "tests/py/dynamo/llm/"],
    ),
    "l2 dynamo core tests": dict(
        fn="trt_tier_l2_dynamo_core",
        paths=["tests/py/dynamo/runtime/", "tests/py/dynamo/executorch/"],
    ),
    "l2 dynamo plugin tests": dict(
        fn="trt_tier_l2_plugin",
        paths=[
            "tests/py/dynamo/conversion/",
            "tests/py/dynamo/automatic_plugin/",
            "tests/py/kernels/",
        ],
    ),
    "l2 torch script tests": dict(
        fn="trt_tier_l2_torchscript", paths=["tests/py/ts/integrations/"]
    ),
    "l2 dynamo distributed tests": dict(
        fn="trt_tier_l2_distributed", paths=["tests/py/dynamo/distributed/"]
    ),
}


def _norm(s: str) -> str:
    """Lowercase and collapse separators so 'L2-dynamo-core-tests' and
    'L2 dynamo core tests' hash to the same tier key."""
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


# ── manifest-backed resolution (group → what ran + how to reproduce) ──────────
def _suite_group_index():
    """Map a normalized CI job group to ``(suite, variant)``.

    CI names each test job ``<suite>-<variant>`` so the parsed group is exactly a
    manifest suite. Index both the ``<suite>-<variant>`` and bare ``<suite>``
    forms; per-variant ``paths`` overrides (e.g. RTX) are applied at lookup."""
    idx = {}
    for s in _SUITES:
        for v in s.variants:
            idx.setdefault(_norm(f"{s.name}-{v}"), (s, v))
        idx.setdefault(_norm(s.name), (s, s.variants[0]))
    return idx


_SUITE_INDEX = _suite_group_index()
# Longest keys first so 'dynamo runtime smoke' wins over 'dynamo runtime'.
_SUITE_KEYS = sorted(_SUITE_INDEX, key=len, reverse=True)


def _suite_repro(suite, variant, kexpr=""):
    """The exact command CI ran for this suite (what `just suite` invokes)."""
    var = f" --variant {variant}" if variant and variant != "standard" else ""
    ka = f' -- -k "{kexpr}"' if kexpr else ""
    return (
        f"TRT_PYTEST_RERUNS=0 uv run --no-sync python -m tests.ci "
        f"run {suite.name}{var}{ka}"
    )


def info_for(group: str):
    """Resolve a CI job group → ``{paths, repro, label}`` from the tests/ci
    manifest, falling back to the legacy ``TIER_MAP`` for pre-migration runs.

    ``paths`` are repo-relative (``cwd`` joined with the suite's pytest
    positionals) so a red cell still points at code. ``repro(kexpr)`` returns the
    local reproduce command, narrowed to the failing tests when ``kexpr`` given.
    """
    g = _norm(group)
    hit = _SUITE_INDEX.get(g) or next(
        (_SUITE_INDEX[k] for k in _SUITE_KEYS if k in g), None
    )
    if hit:
        s, v = hit
        fv = s.for_variant(v)
        cwd = fv["cwd"].rstrip("/")
        paths = [cwd if p in (".", "./") else f"{cwd}/{p}" for p in fv["paths"]]
        paths = paths or [cwd]
        return dict(
            paths=paths,
            repro=lambda kexpr="", _s=s, _v=v: _suite_repro(_s, _v, kexpr),
            label=f"{s.name} · {v}",
        )
    t = TIER_MAP.get(g)
    if t:
        return dict(
            paths=t["paths"],
            repro=lambda kexpr="", _t=t: (
                'source tests/py/utils/ci_helpers.sh && PYTHON="uv run '
                f'--no-sync python" TRT_PYTEST_RERUNS=0 {_t["fn"]}'
                + (f' -k "{kexpr}"' if kexpr else "")
            ),
            label=None,
        )
    return None


# ── gh plumbing (cached) ─────────────────────────────────────────────────────
class Cache:
    """Tiny TTL cache. Completed runs/jobs are immutable so we cache them long;
    anything still in flight gets a short TTL so the UI stays live."""

    def __init__(self):
        self._d: dict[str, tuple[float, object]] = {}
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            hit = self._d.get(key)
        if not hit:
            return None
        expires, val = hit
        return None if time.time() > expires else val

    def put(self, key, val, ttl):
        with self._lock:
            self._d[key] = (time.time() + ttl, val)

    def clear(self):
        with self._lock:
            self._d.clear()


CACHE = Cache()


def gh(args, parse=True, timeout=90):
    proc = subprocess.run(
        ["gh", *args], capture_output=True, text=True, timeout=timeout, cwd=REPO_ROOT
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "gh failed").strip())
    return json.loads(proc.stdout) if parse else proc.stdout


@lru_cache(maxsize=1)
def repo_slug():
    try:
        return gh(
            ["repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
            parse=False,
        ).strip()
    except Exception:
        return "pytorch/TensorRT"


_DEFAULT_BRANCH = None


def default_branch():
    if _DEFAULT_BRANCH:
        return _DEFAULT_BRANCH
    try:
        b = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        ).stdout.strip()
        return b if b and b != "HEAD" else "main"
    except Exception:
        return "main"


# ── test-plan preview: what a branch's labels will actually run ───────────────
# This mirrors, in Python, exactly what CI derives from the triggering event:
#   1. labels → (lane, backend)              — .github/workflows/_decide.yml
#   2. (lane, backend) → per-platform channels — the `if:` gates in the
#      ci-linux-x86_64 / ci-windows / ci-sbsa entry workflows
#   3. (lane, variant, platform) → suites     — `tests.ci matrix` (_test-linux.yml)
# Keep in sync with those files if the gating changes.

# The CI-control labels (the ONLY ones that change what runs) + the axis each
# drives and a one-line effect. Order = display order. Kept in sync with _decide.yml.
CI_LABELS = [
    ("ci: full", "lane", "all tiers (L0–L2) on Linux + Windows"),
    ("ci: nightly", "lane", "everything, incl. llm / kernels / distributed"),
    ("backend: TensorRT", "std", "test the standard TensorRT engine"),
    ("backend: TensorRT-RTX", "rtx", "test the TensorRT-RTX engine"),
]
_CI_LABEL_NAMES = {n for n, _, _ in CI_LABELS}
_INERT_LABEL = "Force All Tests[L0+L1+L2]"  # replaced by ci: full / ci: nightly


def _ci_label_chips(labels, pr):
    """The four CI-control labels as on/off toggles — clicking (then confirming)
    adds/removes the label on the PR live. Disabled when there's no PR to edit.
    (Adding a CI label triggers a run, hence the confirm.)"""
    present = set(labels)
    num = pr["number"] if pr else None
    out = []
    for name, group, effect in CI_LABELS:
        on = name in present
        title = f'{"remove" if on else "add"} — {effect}'
        out.append(
            f'<button class="cilabel {group} {"on" if on else "off"}" data-pr="{num or ""}" '
            f'data-label="{e(name)}" data-add="{"0" if on else "1"}" '
            f'onclick="toggleLabel(this)" title="{e(title)}"{"" if num else " disabled"}>'
            f'<span class="cilmark">{"✓" if on else "+"}</span>{e(name)}</button>'
        )
    return "".join(out)


def resolve_lane_backend(labels, event="pull_request"):
    """labels → (lane, backend), per _decide.yml. `contains()` in Actions is an
    exact array-element match, so the two backend labels never alias."""
    L = set(labels or [])
    has_full, has_nightly = "ci: full" in L, "ci: nightly" in L
    has_rtx, has_std = "backend: TensorRT-RTX" in L, "backend: TensorRT" in L
    if event == "schedule":
        lane = "nightly"
    elif event == "push":  # main canary
        lane = "full"
    else:  # pull_request
        lane = "nightly" if has_nightly else "full" if has_full else "fast"
    if event != "pull_request":
        backend = "both"
    elif has_rtx and has_std:
        backend = "both"
    elif has_rtx:
        backend = "rtx"
    elif has_std:
        backend = "standard"
    elif lane == "fast":  # cheap default on a plain PR push
        backend = "standard"
    else:
        backend = "both"
    return lane, backend


def compute_plan(lane, backend):
    """The channels that will run for (lane, backend), each with its suite list.
    Mirrors the entry workflows' channel `if:` gates. Returns [{platform, engine,
    kind, suites}], suites empty for build-only channels."""
    if _ci_matrix is None:
        return []
    run, fullish = lane != "skip", lane in ("full", "nightly")
    std, rtx = backend != "rtx", backend != "standard"  # which engine channels run
    plan = []

    def add(platform, engine, kind, suite_lane, suite_platform):
        suites = (
            [
                m["suite"]
                for m in _ci_matrix(
                    lane=suite_lane, variant=engine, platform=suite_platform
                )
            ]
            if suite_platform
            else []
        )
        plan.append(dict(platform=platform, engine=engine, kind=kind, suites=suites))

    # Linux x86_64 (ci-linux-x86_64.yml): tests AND python-only on any non-skip lane
    # (incl. fast — PYTHON_ONLY=1 skips Bazel, so it's a cheap per-push smoke).
    if run and std:
        add("Linux x86_64", "standard", "tests", lane, "linux-x86_64")
    if run and rtx:
        add("Linux x86_64", "rtx", "tests", lane, "linux-x86_64")
    if run and std:
        add("Linux x86_64", "standard", "python-only", "python-only", "linux-x86_64")
    if run and rtx:
        add("Linux x86_64", "rtx", "python-only", "python-only", "linux-x86_64")
    # Windows (ci-windows.yml): full/nightly only — never on the fast lane
    if fullish and std:
        add("Windows", "standard", "tests", lane, "windows")
    if fullish and rtx:
        add("Windows", "rtx", "tests", lane, "windows")
    if fullish and std:
        add("Windows", "standard", "python-only", "python-only", "windows")
    if fullish and rtx:
        add("Windows", "rtx", "python-only", "python-only", "windows")
    # SBSA aarch64 (ci-sbsa.yml): BUILD-ONLY (no GPU runners), full/nightly, standard
    if fullish and std:
        add("Linux aarch64 · SBSA", "standard", "build-only", lane, None)
        add(
            "Linux aarch64 · SBSA",
            "standard",
            "build-only · py-only",
            "python-only",
            None,
        )
    return plan


def pr_for(branch, refresh=False):
    """The PR whose head is `branch` — an OPEN one if there is one, else the most
    recent of any state (so a merged/closed PR's number still shows). Fields:
    number/title/url/labels/baseRefName/state, or None. Cached briefly so the board
    banner + plan panel share one lookup (a `{}` sentinel caches the 'no PR' answer)."""
    key = f"pr:{branch}"
    if not refresh and (c := CACHE.get(key)) is not None:
        return c or None
    try:
        prs = gh(
            [
                "pr",
                "list",
                "--head",
                branch,
                "--state",
                "all",
                "--json",
                "number,title,url,labels,baseRefName,state",
                "--limit",
                "5",
            ]
        )
        pr = next((p for p in prs if p.get("state") == "OPEN"), prs[0] if prs else None)
    except Exception:
        pr = None
    CACHE.put(key, pr or {}, ttl=120)
    return pr


def resolve_pr_branch(num):
    """A PR number → its head branch name, so `#4414` typed in the search box (or a
    `?branch=%234414` URL) loads the right branch. Cached; None if no such PR."""
    num = str(num or "").lstrip("#").strip()
    if not re.fullmatch(r"\d+", num):
        return None
    key = f"prbranch:{num}"
    if (c := CACHE.get(key)) is not None:
        return c or None
    try:
        b = gh(
            [
                "pr",
                "view",
                num,
                "--repo",
                repo_slug(),
                "--json",
                "headRefName",
                "-q",
                ".headRefName",
            ],
            parse=False,
        ).strip()
    except Exception:
        b = None
    CACHE.put(key, b or "", ttl=300)
    return b or None


# CI-control comment commands the dashboard can POST to a PR (handled by
# retrigger-ci.yml). (command, effect, is_destructive) — the whitelist is also the
# server-side guard, so only these exact strings can ever be posted.
CI_ACTIONS = [
    ("/rerun", "re-run only the failed / cancelled jobs", False),
    ("/rerun all", "re-run every job from scratch", False),
    ("/cancel", "cancel stale zombie runs on old commits", False),
    ("/cancel all", "cancel EVERY in-flight run for the PR", True),
    ("/test full", "run the full suite (all tiers, both engines)", False),
    ("/test full rtx", "full suite, RTX engine only", False),
    ("/test nightly", "everything incl. llm / kernels / distributed", False),
]
_ALLOWED_CMDS = {c for c, _, _ in CI_ACTIONS}


def post_pr_comment(pr, cmd):
    """Post one whitelisted CI command as a PR comment (which triggers
    retrigger-ci.yml). Raises on a bad PR number or non-whitelisted command —
    the whitelist is the guard against arbitrary comments."""
    if cmd not in _ALLOWED_CMDS:
        raise ValueError(f"command not allowed: {cmd!r}")
    if not re.fullmatch(r"\d+", str(pr or "")):
        raise ValueError(f"bad PR number: {pr!r}")
    return gh(
        ["pr", "comment", str(pr), "--repo", repo_slug(), "--body", cmd], parse=False
    ).strip()


def set_pr_label(pr, label, add):
    """Add or remove one CI-control label on a PR (adding triggers a run via the
    `labeled` event). Whitelisted to the CI labels — the guard against editing
    arbitrary labels. The client reloads with refresh=1 so the plan re-resolves."""
    if label not in _CI_LABEL_NAMES:
        raise ValueError(f"label not allowed: {label!r}")
    if not re.fullmatch(r"\d+", str(pr or "")):
        raise ValueError(f"bad PR number: {pr!r}")
    flag = "--add-label" if add else "--remove-label"
    return gh(
        ["pr", "edit", str(pr), "--repo", repo_slug(), flag, label], parse=False
    ).strip()


# ── data layer ───────────────────────────────────────────────────────────────
def get_runs(branch, limit=40, refresh=False):
    key = f"runs:{branch}:{limit}"
    if not refresh and (c := CACHE.get(key)) is not None:
        return c
    fields = (
        "databaseId,workflowName,displayTitle,headBranch,headSha,status,"
        "conclusion,event,createdAt,updatedAt,url,number"
    )
    runs = gh(
        ["run", "list", "--branch", branch, "--limit", str(limit), "--json", fields]
    )
    newest = {}
    for r in runs:
        r["id"] = r.get("databaseId")  # normalize (gh run list uses databaseId)
        wf = r.get("workflowName") or "?"
        if wf not in newest or r["createdAt"] > newest[wf]["createdAt"]:
            newest[wf] = r
    result = list(newest.values())
    live = any(r["status"] != "completed" for r in result)
    CACHE.put(key, result, ttl=15 if live else 120)
    return result


def _parse_job(j):
    raw = j.get("name", "")
    parts = [p.strip() for p in raw.split(" / ")]
    last = parts[-1]
    py = re.search(r"(?<![\d.])3\.\d+", raw)
    cu = re.search(r"cu\d{2,3}", raw)
    is_matrix = len(parts) > 1 and bool(re.search(r"--|build-wheel|3\.\d+", last))
    group_parts = parts[:-1] if is_matrix else parts[:]
    if group_parts and group_parts[0] in ("core", "build"):
        rest = group_parts[1:]
        group_parts = rest if rest else (parts[:-1] if is_matrix else parts)
    group = " / ".join(group_parts) if group_parts else raw
    gl = _norm(group)
    last_word = gl.rsplit(" ", 1)[-1] if gl else ""
    if "build-wheel" in last or gl.startswith("build "):
        kind = "build"
    elif (
        re.search(r"\bl[012]\b", gl)  # tier token (old `L2 …` names, new-format tier)
        or " / test (" in group  # new `<channel> / test (…)` format
        or last_word in ("test", "tests")  # descriptive `… dynamo runtime tests` groups
    ):
        kind = "test"
    elif "matrix" in gl or "generate" in gl or group == "filter-matrix":
        kind = "setup"
    elif (
        raw.startswith("CI /")
        or "aggregate" in gl
        or "collect results" in gl
        or "rollup" in gl
    ):
        kind = "rollup"
    else:
        kind = "other"
    return dict(
        id=j.get("databaseId") or j.get("id"),
        raw=raw,
        group=group,
        kind=kind,
        python=py.group(0) if py else None,
        cuda=cu.group(0) if cu else None,
        status=j.get("status"),
        conclusion=j.get("conclusion"),
        url=j.get("html_url") or j.get("url"),
        failedSteps=[
            s["name"] for s in j.get("steps", []) if s.get("conclusion") == "failure"
        ],
    )


def get_jobs(run_id, refresh=False):
    key = f"jobs:{run_id}"
    if not refresh and (c := CACHE.get(key)) is not None:
        return c
    data = gh(
        [
            "api",
            f"repos/{repo_slug()}/actions/runs/{run_id}/jobs",
            "--paginate",
            "-q",
            ".jobs[]",
        ],
        parse=False,
    )
    jobs = [json.loads(line) for line in data.splitlines() if line.strip()]
    parsed = [_parse_job(j) for j in jobs]
    live = any(j["status"] != "completed" for j in parsed)
    CACHE.put(key, parsed, ttl=15 if live else 300)
    return parsed


NOISE = re.compile(
    r"Process completed with exit code|Node\.js \d+ is deprecated|"
    r"is deprecated\. Please switch|conda\.cli|defaults' channel|auto_activate|"
    r"might have been added implicitly",
    re.I,
)
# A test id is the annotation's first line: a single token that names a test —
# a bare `test_x[param]`, a dotted `Class.test_x`, or a `file.py::Class::test_x`
# nodeid. (The old regex required a dot, so it missed bare parametrized names.)
TESTLINE = re.compile(r"^[\w./\[\]:-]+$")


def _test_name(head):
    """Pull a test id out of an annotation's first line, or None."""
    if not head or " " in head or not TESTLINE.match(head):
        return None
    tok = head.split("::")[-1] if "::" in head else head  # nodeid → last segment
    return tok if re.search(r"test", tok, re.I) else None


def _annot_loc(a):
    """Exact (repo-relative file, line) straight from the annotation — better than
    grep, which only finds the def line. The path is prefixed (pytorch/tensorrt/…);
    keep from `tests/` on and confirm it exists locally."""
    m = re.search(r"(tests/.+)$", a.get("path") or "")
    if not m or not os.path.exists(os.path.join(REPO_ROOT, m.group(1))):
        return None
    return (m.group(1), a.get("start_line") or 0)


def _git_grep(pat):
    try:
        out = subprocess.run(
            ["git", "grep", "-n", "-E", pat, "--", "tests/"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=15,
        ).stdout
    except Exception:
        return None
    for line in out.splitlines():
        path, lineno, _ = line.split(":", 2)
        return (path, int(lineno))
    return None


@lru_cache(maxsize=4096)
def grep_test(symbol):
    """Resolve a failing-test symbol (Class.method / module.func) to (file, line)
    via `git grep`. Tries the method, then the enclosing class — parametrized names
    (`@parameterized.expand` → test_x_5) have no literal `def`, so the class is the
    best anchor. Returns None if nothing matches."""
    parts = [re.sub(r"\[.*$", "", p) for p in re.split(r"[.:]+", symbol.strip()) if p]
    for token in reversed(parts):  # method first, then its class
        if not re.match(r"^[A-Za-z_]\w+$", token):
            continue
        pat = f"def {token}\\b" if token.startswith("test") else f"class {token}\\b"
        if hit := _git_grep(pat):
            return hit
    return None


def get_failures(job_id, refresh=False):
    key = f"fail:{job_id}"
    if not refresh and (c := CACHE.get(key)) is not None:
        return c
    try:
        anns = gh(["api", f"repos/{repo_slug()}/check-runs/{job_id}/annotations"])
    except Exception as e:
        return dict(error=str(e), failures=[])
    seen, failures = set(), []
    for a in anns:
        if a.get("annotation_level") != "failure":
            continue
        msg = (a.get("message") or "").strip()
        if not msg or NOISE.search(msg.splitlines()[0]):
            continue
        head = msg.splitlines()[0].strip()
        test = _test_name(head)
        dedupe = test or head
        if dedupe in seen:
            continue
        seen.add(dedupe)
        loc = _annot_loc(a) or (
            grep_test(test) if test else None
        )  # annotation line first
        failures.append(
            dict(
                test=test,
                message=msg[:1400],
                file=loc[0] if loc else None,
                line=loc[1] if loc else None,
            )
        )
    result = dict(failures=failures)
    CACHE.put(key, result, ttl=600)
    return result


# ── job logs (fetch + per-test extraction) ──────────────────────────────────
_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z ")  # GH line prefix
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")  # SGR / CSI colour codes
_GROUP = re.compile(r"^##\[(?:group|section|command)\]")  # fold marker → keep title
_ENDGROUP = re.compile(r"^##\[endgroup\]\s*$")  # pure noise
_SEP = re.compile(r"^_{4,}\s+(.+?)\s+_{4,}\s*$")  # pytest FAILURES sep
_SECT = re.compile(r"^={4,}")  # ==== section ====


def get_job_log(job_id, refresh=False):
    """Raw plain-text log for one job (immutable once complete → cached 1h)."""
    key = f"log:{job_id}"
    if not refresh and (c := CACHE.get(key)) is not None:
        return c
    try:
        raw = gh(
            ["api", f"repos/{repo_slug()}/actions/jobs/{job_id}/logs"],
            parse=False,
            timeout=60,
        )
    except Exception:
        raw = None
    CACHE.put(key, raw, ttl=3600 if raw else 20)
    return raw


def _clean_lines(raw):
    """Strip the GH timestamp prefix, ANSI colour codes, and the ##[group] /
    ##[endgroup] fold markers that otherwise leak into the <pre>."""
    out = []
    for ln in raw.splitlines():
        ln = _ANSI.sub("", _TS.sub("", ln)).replace("\r", "")
        if _ENDGROUP.match(ln):
            continue
        out.append(_GROUP.sub("", ln))
    return out


def extract_test_log(lines, test, max_lines=160):
    """Pull the pytest FAILURES block for `test`. Falls back to a context window
    around the test's last mention (covers custom harnesses w/o a FAILURES sep).
    Returns the excerpt string, or None."""
    leaf = re.split(r"[.:]", test)[-1].split("[")[0] if test else None
    seps = [
        (i, m.group(1).strip()) for i, ln in enumerate(lines) if (m := _SEP.match(ln))
    ]
    start = None
    if test:
        # Prefer an exact title match — the leaf method name (e.g. test_trt_compile)
        # is shared across many classes, so fuzzy matching would grab the wrong block.
        for i, title in seps:
            if title == test or title.split("[")[0] == test:
                start = i
                break
        if start is None and leaf:  # fall back when the annotation gave only a leaf
            for i, title in seps:
                base = title.split("[")[0]
                if base == leaf or base.endswith("." + leaf):
                    start = i
                    break
    if start is not None:
        nexts = [i for i, _ in seps if i > start] + [len(lines)]
        end = min(nexts)
        for j in range(start + 1, end):
            if _SECT.match(lines[j]):
                end = j
                break
        block = lines[start:end]
    else:
        needle = leaf or test
        hit = next(
            (i for i in range(len(lines) - 1, -1, -1) if needle and needle in lines[i]),
            None,
        )
        if hit is None:
            return None
        block = lines[max(0, hit - 4) : hit + 44]
    if len(block) > max_lines:
        block = block[:max_lines] + [
            f"… (+{len(block) - max_lines} more lines — open the raw log)"
        ]
    return "\n".join(block).strip()


# ── anchoring: find the *real* error in a non-pytest (build/install) failure ──
# In GH logs the first `##[error]` is almost always echoed shell text
# (`echo "::error::…"`), so we never anchor on it. The reliable signals, in order:
# the pytest summary, a pip/build metadata error, or a walk backward from the
# `Process completed with exit code N` line to the nearest genuine error.
_SUMMARY_HDR = re.compile(r"^=+\s*short test summary info\s*=+", re.I)
_PIP_ERR = re.compile(
    r"error: subprocess-exited-with-error|did not run successfully|^\s*╰─>"
)
_EXITCODE = re.compile(r"(?:Process completed with exit code|exit code:?)\s*[1-9]")
_ECHO_NOISE = re.compile(
    r'echo\s+["\']?::|Specified script file|^\s*(?:if|else|elif|fi|then|source|for|do|done)\b'
)
_REAL_ERR = re.compile(
    r"Traceback \(most recent call last\)|\b\w*(?:Error|Exception)\b\s*:|"
    r"^E\s{2,}\S|^\s*(?:FAILED|ERROR)\s|error: subprocess-exited-with-error|"
    r"CMake Error|fatal error:|ninja: build stopped|Segmentation fault|OutOfMemory|Killed\b"
)


def find_error_window(lines, max_lines=140):
    """Anchor a non-pytest failure on its real error instead of the blind tail.
    Returns (title, start, end)."""
    n = len(lines)
    if not n:
        return ("end of job log", 0, 0)
    for i, ln in enumerate(lines):  # 1. pytest summary
        if _SUMMARY_HDR.search(ln):
            end = next(
                (
                    j
                    for j in range(i + 1, n)
                    if _SECT.match(lines[j]) and not _SUMMARY_HDR.search(lines[j])
                ),
                n,
            )
            return ("short test summary", i, min(end, i + max_lines))
    for i, ln in enumerate(lines):  # 2. pip / build error
        if _PIP_ERR.search(ln):
            return ("install / build error", max(0, i - 3), min(n, i + max_lines))
    exit_i = next(
        (i for i in range(n - 1, -1, -1) if _EXITCODE.search(lines[i])), n - 1
    )
    anchor = next(
        (
            i
            for i in range(exit_i, -1, -1)  # 3. walk back to real error
            if _REAL_ERR.search(lines[i]) and not _ECHO_NOISE.search(lines[i])
        ),
        None,
    )
    if anchor is not None:
        start = anchor
        for j in range(
            anchor, max(0, anchor - 80), -1
        ):  # widen up to the traceback head
            if "Traceback (most recent call last)" in lines[j]:
                start = j
                break
        return ("error", max(0, start - 2), min(n, exit_i + 1))
    return ("end of job log", max(0, n - 180), n)  # 4. fallback: tail


# ── root-cause collapse: many identical failures → one line + a pointer ───────
_EXC_RE = re.compile(r"\b([A-Z]\w*(?:Error|Exception|Warning))\b\s*:?\s*([^\n]*)")


def _reason_key(msg):
    """Stable, comparable essence of a failure message, for de-duplication.
    Volatile bits (quoted names, addresses, numbers, paths) are stripped so
    parametrized variants of one failure collapse together."""
    m = _EXC_RE.search(msg or "")
    if m:
        detail = re.sub(r"['\"][^'\"]*['\"]|0x[0-9a-fA-F]+|\d+|/\S+", "", m.group(2))
        detail = re.sub(r"\s+", " ", detail).strip()
        return f"{m.group(1)}: {detail}"[:90].rstrip(": ")
    head = next((l.strip() for l in (msg or "").splitlines() if l.strip()), "")
    return re.sub(r"\d+", "", head)[:90]


def _reason_first(msg):
    """The single most informative line of a message, for a block subtitle."""
    m = _EXC_RE.search(msg or "")
    if m:
        return m.group(0)[:300]
    return next((l.strip() for l in (msg or "").splitlines() if l.strip()), "")[:300]


def _group_failures(fails):
    """Group failures by normalized reason, biggest group first (a systemic
    break surfaces above one-offs). Returns [(reason, [failure, …]), …]."""
    groups, order = {}, []
    for f in fails:
        k = _reason_key(f["message"])
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(f)
    return sorted(((k, groups[k]) for k in order), key=lambda kv: -len(kv[1]))


# ── severity rendering: per-line spans so the client can search / jump / colour ─
_SEV_ERR = re.compile(
    r"^##\[error\]|Traceback \(most recent call last\)|\b\w*(?:Error|Exception)\b\s*:|"
    r"^E\s{2,}\S|^\s*(?:FAILED|ERROR)\s|error: subprocess-exited-with-error|"
    r"CMake Error|fatal error:|Segmentation fault|\bAssertionError\b|exit code:?\s*[1-9]"
)
_SEV_WARN = re.compile(r"^##\[warning\]|\bwarning\b|\bdeprecat|\bWARN\b", re.I)
_SEV_PASS = re.compile(r"\bPASSED\b|=+\s*\d+ passed|^\s*ok\s*$", re.I)


def _render_logtext(text):
    """Emit log text as one <span class=ll> per line, tagged by severity.
    Returns (html, error_hits) where error_hits = [(line_index, short_text)]
    for the jump-list."""
    spans, hits = [], []
    for i, ln in enumerate(text.split("\n")):
        sev = ""
        if _SEV_ERR.search(ln) and not _ECHO_NOISE.search(ln):
            sev, _ = "err", hits.append((i, ln.strip()[:90] or "error"))
        elif _SEV_WARN.search(ln):
            sev = "warn"
        elif _SEV_PASS.search(ln):
            sev = "pass"
        attr = f' data-sev="{sev}"' if sev else ""
        spans.append(f'<span class="ll"{attr}>{e(ln) or "&nbsp;"}</span>')
    return "".join(spans), hits  # .ll is display:block; no newlines between


def _logblock(title, subtitle, excerpt, is_open=False):
    """A collapsible log excerpt: severity-highlighted <pre> + an error jump-list."""
    if not excerpt:
        return (
            f'<details class="logblock"><summary>{e(title)}</summary>{subtitle}'
            '<div class="muted" style="padding:8px 12px 12px">No matching block '
            "in the log — see the raw log.</div></details>"
        )
    body, hits = _render_logtext(excerpt)
    jumps = ""
    if hits:
        chips = "".join(
            f'<button class="jump" data-line="{i}" onclick="jumpLine(this)" '
            f'title="{e(short)}">{e(short[:44])}</button>'
            for i, short in hits[:8]
        )
        jumps = f'<div class="logjumps"><span class="lbl">jump</span>{chips}</div>'
    op = " open" if is_open else ""
    return (
        f'<details class="logblock"{op}><summary>{e(title)}</summary>{subtitle}'
        f'{jumps}<pre class="logtext">{body}</pre></details>'
    )


_IMPORT_FAIL = re.compile(
    r"ModuleNotFoundError|ImportError|No module named|"
    r"error collecting|during collection|failed to import",
    re.I,
)


def _install_error_block(lines):
    """Locate a pip/build install error in the log and render it, or return ''."""
    pin = next((i for i, ln in enumerate(lines) if _PIP_ERR.search(ln)), None)
    if pin is None:
        return ""
    body, _h = _render_logtext("\n".join(lines[max(0, pin - 3) : pin + 40]).strip())
    return (
        '<details class="logblock rootcause" open><summary>'
        "likely root cause · install / build error</summary>"
        f'<pre class="logtext">{body}</pre></details>'
    )


def _root_cause_banner(lines, fails, groups):
    """When most failures share one reason, say so once and point at the cause."""
    reason, members = groups[0]
    n = len(members)
    msg = (
        f"<strong>{n} of {len(fails)}</strong> failures share one cause: "
        f"<code>{e(reason)}</code>."
    )
    extra = ""
    if _IMPORT_FAIL.search(reason):
        msg += (
            f" That is an <em>install / build</em> failure, not {n} test bugs — "
            "the package never imported."
        )
        extra = _install_error_block(lines)
    return f'<div class="rootbanner">{msg}</div>{extra}'


def render_joblog(job_id, url):
    log = get_job_log(job_id)
    ghlink = (
        f'<a href="{e(url)}" target="_blank" rel="noopener">job on GitHub ↗</a>'
        if url
        else ""
    )
    rawlink = f'<a href="/ui/joblog?job={e(job_id)}&raw=1" target="_blank" rel="noopener">raw log ↗</a>'
    head = f'<div class="logsec-head"><span class="lbl">relevant logs</span>{rawlink} · {ghlink}</div>'
    if not log:
        return head + (
            '<div class="muted">Log not available yet — the job may still be '
            f"running, or GitHub expired it. {ghlink}</div>"
        )
    lines = _clean_lines(log)
    tools = (
        '<div class="logtools">'
        '<input class="logfind" type="search" placeholder="search these logs…" '
        'oninput="logSearch(this)" aria-label="search logs">'
        '<label class="logfilt"><input type="checkbox" onchange="logFilter(this)"> matches only</label>'
        '<span class="logcount muted"></span></div>'
    )
    fails = get_failures(job_id).get("failures", [])
    blocks = []
    if fails:
        groups = _group_failures(fails)
        if len(fails) >= 3 and any(len(m) > 1 for _, m in groups):
            blocks.append(_root_cause_banner(lines, fails, groups))
        for reason, members in groups:
            if len(members) == 1:
                f = members[0]
                sub = f'<div class="logreason">{e(_reason_first(f["message"]))}</div>'
                blocks.append(
                    _logblock(
                        f["test"] or "failure",
                        sub,
                        extract_test_log(lines, f["test"]),
                        is_open=True,
                    )
                )
            else:  # collapsed: N parametrized / duplicate failures → one block
                names = ", ".join(m["test"] or "?" for m in members[:10])
                more = f" +{len(members) - 10} more" if len(members) > 10 else ""
                sub = (
                    f'<div class="logreason">{e(reason)}</div>'
                    f'<div class="logmeta">{len(members)} tests · {e(names)}{e(more)}</div>'
                )
                blocks.append(
                    _logblock(
                        f"{len(members)}× {reason}",
                        sub,
                        extract_test_log(lines, members[0]["test"]),
                        is_open=True,
                    )
                )
    else:  # no annotations → anchor on the real error; surface install failures
        title, s, en = find_error_window(lines)
        window = "\n".join(lines[s:en]).strip()
        import_dominated = (
            title == "short test summary" and len(_IMPORT_FAIL.findall(window)) >= 2
        )
        inst = _install_error_block(lines) if import_dominated else ""
        if inst:
            blocks.append(inst)
            blocks.append(
                '<div class="rootbanner">Every test failed at <em>import</em> — this is '
                "an <em>install / build</em> failure (above), not test bugs.</div>"
            )
        blocks.append(_logblock(title, "", window, is_open=not inst))
    return head + tools + "".join(blocks)


# ── status helpers ───────────────────────────────────────────────────────────
def classify(status, conclusion):
    """→ (css class, label). "Didn't run" states (skipped/cancelled/never-started)
    are kept DISTINCT from "fail" — a gated-off or blocked tier is not a failure,
    and a cancelled job didn't produce a verdict. Only `failure`/`timed_out`
    (actually executed and failed) are `fail`."""
    if status and status != "completed":
        if status in ("in_progress", "running"):
            return "run", "running"
        return "queue", status.replace(
            "_", " "
        )  # queued / waiting / pending / requested
    c = conclusion or ""
    return {
        "success": ("pass", "passed"),
        "failure": ("fail", "failed"),
        "timed_out": ("fail", "timed out"),
        # ── didn't run (by design / gating / blocked by a failed dependency) ──
        "skipped": ("skip", "didn’t run"),
        "neutral": ("skip", "neutral"),
        "stale": ("skip", "stale"),
        # ── didn't finish / never started (distinct again from a test failure) ──
        "cancelled": ("cancel", "cancelled"),
        "startup_failure": ("cancel", "didn’t start"),
        "action_required": ("queue", "action req"),
    }.get(c, ("skip", c or "no result"))


# Sort worst-first: real failures, then in-flight, then didn't-finish, then green,
# then the didn't-run noise last. "skip" (gated-off) is deliberately below "pass".
RANK = {"fail": 0, "run": 1, "queue": 2, "cancel": 3, "pass": 4, "skip": 5}
DIDNT_RUN = ("skip", "cancel")


# ── failure categorization ───────────────────────────────────────────────────
# Triage-at-a-glance: the judgment a human makes reading each traceback — is this
# infra (OOM/resource → retry/parallelism, not a code bug), an env/import gap, a
# converter gap, or a real regression (numerical/assertion)? First pattern wins,
# so order is most-specific → most-generic. `kind` drives the tag colour:
#   infra = "probably not your bug"  ·  env/gap = setup/feature hole  ·  bug = real.
# NOTE: an OOM often surfaces AS an AssertionError ("assert cuda_engine"); the OOM
# signatures below are listed first on purpose so they win over the generic assert.
_FAIL_CATS = [
    (
        "oom",
        "GPU / resource",
        "infra",
        re.compile(
            r"OutOfMemory|out of memory|createCaskHardwareInfo|\bCask\b|"
            r"build_serialized_network returned None|assert cuda_engine|cuda_engine\b|"
            r"catchCudaError|cudaError|cuInit|defaultAllocator|no CUDA GPUs|"
            r"CUDA error|cudaMalloc|device-side assert",
            re.I,
        ),
    ),
    (
        "timeout",
        "timeout",
        "infra",
        re.compile(r"timed out|timeout|deadline exceeded|took too long", re.I),
    ),
    (
        "import",
        "import / collection",
        "env",
        re.compile(
            r"ModuleNotFoundError|ImportError|No module named|cannot import name|"
            r"error collecting|errors during collection|failed to import",
            re.I,
        ),
    ),
    (
        "unsupported",
        "unsupported op",
        "env",
        re.compile(
            r"no converter|not support|unsupported|Could not find any implementation|"
            r"UnsupportedOperator|no implementation for",
            re.I,
        ),
    ),
    (
        "numerical",
        "numerical / accuracy",
        "bug",
        re.compile(
            r"not close|Mismatched elements|cosine|tolerance|allclose|"
            r"Max absolute difference|relative difference|accuracy|atol|rtol",
            re.I,
        ),
    ),
    (
        "assertion",
        "assertion",
        "bug",
        re.compile(r"\bAssertionError\b|^\s*assert\s", re.I | re.M),
    ),
    (
        "typeerr",
        "type / shape",
        "bug",
        re.compile(
            r"\b(TypeError|AttributeError|KeyError|IndexError|ValueError)\b|"
            r"shape mismatch|size mismatch|dtype",
            re.I,
        ),
    ),
    (
        "runtime",
        "runtime error",
        "bug",
        re.compile(r"\b(RuntimeError|NotImplementedError)\b", re.I),
    ),
]


# Roll-up labels for the per-run category tally (kind → human summary word).
_KIND_LABEL = {
    "bug": "real bug",
    "env": "env / gap",
    "infra": "infra / OOM",
    "unknown": "uncategorized",
}


def categorize_failure(message):
    """Classify a failure's error text → {key, label, kind}. `kind` ∈
    {infra, env, bug, unknown} and drives the tag colour."""
    text = message or ""
    for key, label, kind, rx in _FAIL_CATS:
        if rx.search(text):
            return dict(key=key, label=label, kind=kind)
    return dict(key="other", label="error", kind="unknown")


def _cat_tag(cat):
    return (
        f'<span class="cat {cat["kind"]}" '
        f'title="{e(cat["label"])} — {cat["kind"]}">{e(cat["label"])}</span>'
    )


def _config_pattern(occ, all_pys, all_cudas):
    """Which configuration a failure correlates with — the strongest diagnostic
    signal after the category. Only asserts "X only" when X is a strict subset of
    what actually ran (so we never claim a pattern that isn't real)."""
    fpys = {o["py"] for o in occ if o.get("py")}
    fcudas = {o["cuda"] for o in occ if o.get("cuda")}
    fplats = {o["plat"] for o in occ}
    bits = []
    if fplats and all("RTX" in p or "rtx" in p for p in fplats):
        bits.append("RTX only")
    if fplats and all("py-only" in p for p in fplats):
        bits.append("python-only")
    if len(all_cudas) > 1 and len(fcudas) == 1 and fcudas < all_cudas:
        bits.append(f"{next(iter(fcudas))} only")
    if len(all_pys) > 1 and len(fpys) == 1 and fpys < all_pys:
        bits.append(f"py{next(iter(fpys))} only")
    return bits


def rel_time(iso):
    if not iso:
        return ""
    try:
        t = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except Exception:
        return iso
    s = (datetime.now(timezone.utc) - t).total_seconds()
    if s < 60:
        return "just now"
    for unit, n in (("d", 86400), ("h", 3600), ("m", 60)):
        if s >= n:
            return f"{int(s // n)}{unit} ago"
    return "just now"


def pretty_platform(name):
    n = (
        name.replace("Python-only build and test ", "py-only ")
        .replace("RTX - Build and test ", "RTX ")
        .replace("RTX - Python-only build and test ", "RTX py-only ")
        .replace("Build and test ", "")
        .replace(" wheels", "")
        .replace(" for Jetpack", " · jetpack")
    )
    return n.strip()


def e(s):
    return html.escape(str(s if s is not None else ""))


def qp(**kw):
    """Build a URL-encoded, HTML-attribute-safe query string from kwargs, so
    values with spaces/&/ (platform + tier names, job URLs) survive intact."""
    return e(
        "&".join(
            f"{k}={quote(str(v if v is not None else ''), safe='')}"
            for k, v in kw.items()
        )
    )


# ── HTML fragment renderers ──────────────────────────────────────────────────
def _summary_html(runs, oob=False):
    """The top counts strip. Shared by the board and the live poller so the two
    never drift. 'didn't run' folds skipped + cancelled together and is counted
    separately from 'failing'."""
    counts = {"pass": 0, "fail": 0, "run": 0, "queue": 0, "skip": 0, "cancel": 0}
    for r in runs:
        cls, _ = classify(r["status"], r["conclusion"])
        counts[cls] = counts.get(cls, 0) + 1
    didnt_run = counts["skip"] + counts["cancel"]
    head = runs[0] if runs else {}
    full_sha = head.get("headSha") or ""
    sha = full_sha[:9]
    msg = head.get("displayTitle", "")
    sha_html = (
        f'<a class="sha" href="https://github.com/{repo_slug()}/commit/{e(full_sha)}" '
        f'target="_blank" rel="noopener"><code>{e(sha)}</code></a>'
        if full_sha
        else f"<code>{e(sha)}</code>"
    )
    commit = (
        f'<div class="commit" data-sha="{e(sha)}" data-msg="{e(msg)}">'
        f"{sha_html} · {e(msg)[:80]}"
        f' · {e(rel_time(head.get("createdAt")))}</div>'
    )

    def stat(cls, n, word):
        return (
            (
                f'<span class="stat {cls}"><span class="dot {cls}"></span>'
                f'<span class="n">{n}</span> {word}</span>'
            )
            if n
            else ""
        )

    oob_attr = ' hx-swap-oob="true"' if oob else ""
    return (
        f'<div id="summary" class="summary"{oob_attr}>'
        f'{stat("fail", counts["fail"], "failing")}'
        f'{stat("run", counts["run"], "running")}'
        f'{stat("queue", counts["queue"], "queued")}'
        f'{stat("pass", counts["pass"], "passing")}'
        f'{stat("skip", didnt_run, "didn’t run")}'
        f"{commit}</div>"
    )


def _plan_hint(lane, backend):
    if lane == "fast":
        h = (
            "<code>fast</code> = L0 smoke + python-only on Linux x86_64. Add <code>ci: full</code> "
            "for all tiers on Linux + Windows, or <code>ci: nightly</code> to also run "
            "llm / kernels / distributed."
        )
    elif lane == "full":
        h = (
            "<code>full</code> runs L0–L2. Add <code>ci: nightly</code> to also run "
            "llm / kernels / distributed."
        )
    else:
        h = "<code>nightly</code> runs everything."
    if backend == "standard":
        h += " Add <code>backend: TensorRT-RTX</code> to also test RTX."
    elif backend == "rtx":
        h += " Add <code>backend: TensorRT</code> to also test standard."
    return f'<span class="plan-hint">{h}</span>'


def render_plan(branch, refresh=False):
    """'What will run' panel: branch labels → lane/backend → the exact per-platform
    channels + suites CI will execute. Cached briefly (labels change rarely)."""
    key = f"plan:{branch}"
    if not refresh and (c := CACHE.get(key)) is not None:
        return c
    try:
        html = _render_plan(branch)
    except Exception as ex:  # best-effort — never break the board
        html = (
            f'<div class="muted" style="padding:4px 0">plan unavailable: {e(ex)}</div>'
        )
    CACHE.put(key, html, ttl=120)
    return html


def _render_plan(branch):
    pr = None
    if branch == "main":  # the push canary — not any head=main PR
        labels, event, src = [], "push", "push → main"
    elif pr := pr_for(branch):
        labels = [l.get("name", "") for l in pr.get("labels", [])]
        event, src = "pull_request", f'PR #{pr["number"]}'
    else:
        labels, event, src = [], "pull_request", "no open PR"
    lane, backend = resolve_lane_backend(labels, event)
    plan = compute_plan(lane, backend)
    njobs = sum(len(c["suites"]) for c in plan)

    srchtml = (
        f'<a href="{e(pr["url"])}" target="_blank" rel="noopener">{e(src)} ↗</a>'
        if pr
        else e(src)
    )
    # CI-control labels as on/off state; the rest shown muted as "other".
    if event == "push":
        labelblock = (
            '<div class="plan-labels"><span class="muted">push to <code>main</code> — '
            "runs the full lane on both engines regardless of labels.</span></div>"
        )
    else:
        others = [x for x in labels if x not in _CI_LABEL_NAMES and x != _INERT_LABEL]
        inert = (
            f'<button class="cilabel inert" disabled title="inert — superseded by '
            f'ci: full / ci: nightly">⚠ {e(_INERT_LABEL)}</button>'
            if _INERT_LABEL in labels
            else ""
        )
        other_html = (
            (
                '<div class="plan-other"><span class="muted">other:</span> '
                + " ".join(f'<span class="plabel">{e(x)}</span>' for x in others)
                + "</div>"
            )
            if others
            else ""
        )
        labelblock = (
            f'<div class="ci-labels"><span class="cil-head">CI labels</span>'
            f"{_ci_label_chips(labels, pr)}{inert}</div>"
            f'<div class="plan-hint-row">{_plan_hint(lane, backend)}</div>{other_html}'
        )

    # Live CI-control buttons: post a command as a PR comment (with a click-to-confirm
    # guard so a stray click can't trigger a run). Only when there's a PR to post to.
    actions_html = ""
    if pr:
        acts = "".join(
            f'<button class="ciact{" danger" if danger else ""}" data-pr="{pr["number"]}" '
            f'data-cmd="{e(cmd)}" onclick="postCmd(this)" title="{e(desc)} — click, then confirm">'
            f"{e(cmd)}</button>"
            for cmd, desc, danger in CI_ACTIONS
        )
        actions_html = (
            f'<div class="ci-actions"><span class="cil-head">CI actions</span>{acts}'
            f'<span class="ciact-note muted">click, then confirm — posts to '
            f'PR #{pr["number"]}</span></div>'
        )
    labelblock += actions_html
    rows = []
    for c in plan:
        if c["suites"]:
            detail, cnt = (
                f'<span class="psuites">{e(", ".join(c["suites"]))}</span>',
                str(len(c["suites"])),
            )
        else:
            detail, cnt = '<span class="muted">wheel build · no tests</span>', "—"
        rows.append(
            f'<tr><td class="pplat">{e(c["platform"])}</td>'
            f'<td><span class="peng {e(c["engine"])}">{e(c["engine"])}</span></td>'
            f'<td class="pkind">{e(c["kind"])}</td>'
            f'<td class="pcnt">{cnt}</td><td class="pdet">{detail}</td></tr>'
        )
    table = (
        (
            '<table class="plan-table"><thead><tr><th>platform</th><th>engine</th>'
            "<th>kind</th><th>#</th><th>suites</th></tr></thead>"
            f'<tbody>{"".join(rows)}</tbody></table>'
        )
        if rows
        else '<div class="muted">Manifest not available on this checkout.</div>'
    )
    summary = (
        f'<summary><span class="lbl">test plan</span>'
        f'<span class="pverdict">lane <b>{e(lane)}</b> · backend <b>{e(backend)}</b></span>'
        f'<span class="muted">{njobs} test jobs · {srchtml}</span></summary>'
    )
    return (
        f'<details class="plan" open>{summary}'
        f'<div class="plan-body">{labelblock}{table}</div></details>'
    )


def render_board(branch, refresh=False):
    try:
        runs = get_runs(branch, refresh=refresh)
    except Exception as ex:
        return f'<div class="empty-state">Could not load <code>{e(branch)}</code>:<br><br>{e(ex)}</div>'
    if not runs:
        return f'<div class="empty-state">No CI runs found for <code>{e(branch)}</code>.</div>'

    for r in runs:
        r["_cls"], r["_label"] = classify(r["status"], r["conclusion"])
    runs.sort(
        key=lambda r: (
            RANK.get(r["_cls"], 9),
            pretty_platform(r["workflowName"]).lower(),
        )
    )
    summary = _summary_html(runs)

    cards = []
    for r in runs:
        rid, cls, label = r["id"], r["_cls"], r["_label"]
        plat = pretty_platform(r["workflowName"])
        sub = f'{e(r.get("event",""))} · #{r.get("number","")} · {e(rel_time(r.get("createdAt")))}'
        q = qp(
            run=rid,
            sha=r.get("headSha", ""),
            platform=plat,
            refresh="1" if refresh else "0",
        )
        cards.append(f"""
    <details class="card {'fail' if cls=='fail' else ''}"
             hx-get="/ui/platform?{q}" hx-trigger="toggle once"
             hx-target="find .card-body" hx-swap="innerHTML">
      <summary class="card-head">
        <span class="dot {cls}"></span>
        <span class="card-title">{e(plat)}<div class="sub">{sub}</div></span>
        <span id="badge-{rid}" class="card-badge {cls}">{e(label)}</span>
        <span class="caret">▸</span>
      </summary>
      <div class="card-body"><div class="muted" style="padding:10px 0"><span class="spinner"></span> loading jobs…</div></div>
    </details>""")

    poller = (
        f'<div hx-get="/ui/status?branch={e(branch)}" hx-trigger="load delay:20s, every 25s" '
        f'hx-swap="none"></div>'
    )
    agg = (
        f'<h2 class="sec">Failures across platforms</h2>'
        f'<div id="agg" hx-get="/ui/aggregate?branch={e(branch)}" hx-trigger="load" hx-swap="innerHTML">'
        f'<div class="agg-loading"><span class="spinner"></span> scanning failing tests…</div></div>'
    )
    board = f'<h2 class="sec">Platforms</h2><div id="board" class="board">{"".join(cards)}</div>'
    # main is the push canary, not a PR (ignore any stray head=main fork PR) — matches
    # render_plan. pr_for still warms the cache render_plan reuses.
    pr = pr_for(branch, refresh=refresh) if branch != "main" else None
    banner = ""
    if pr:
        base = pr.get("baseRefName", "")
        state = (pr.get("state") or "").upper()
        state_tag = (
            f'<span class="pr-state {state.lower()}">{e(state.lower())}</span>'
            if state and state != "OPEN"
            else ""
        )
        base_html = f'<span class="pr-base">→ {e(base)}</span>' if base else ""
        banner = (
            f'<div class="pr-banner" data-pr="{pr["number"]}" data-base="{e(base)}">'
            f'<a class="pr-num" href="{e(pr["url"])}" target="_blank" '
            f'rel="noopener">PR #{pr["number"]} ↗</a>{state_tag}{base_html}'
            f'<span class="pr-title">{e(pr.get("title", ""))}</span></div>'
        )
    return (
        banner + summary + render_plan(branch, refresh=refresh) + agg + board + poller
    )


def render_status(branch):
    """OOB-only fragment: refresh the summary counts + every card badge in place
    without disturbing expanded platform grids."""
    try:
        runs = get_runs(branch, refresh=True)
    except Exception:
        return ""
    spans = []
    for r in runs:
        cls, label = classify(r["status"], r["conclusion"])
        spans.append(
            f'<span id="badge-{r["id"]}" hx-swap-oob="true" '
            f'class="card-badge {cls}">{e(label)}</span>'
        )
    return _summary_html(runs, oob=True) + "".join(spans)


KIND_ORDER = {"test": 0, "build": 1, "setup": 2, "rollup": 3, "other": 4}


def fail_kind(job):
    """A FAILED test/build job → 'test' (it produced pytest failure annotations, i.e.
    real test failures) or 'infra' (build/setup/env/OOM/runner died with no test
    verdict). Build failures are always infra; test jobs are probed via their (cached)
    annotations. Lets the grid show a broken runner differently from a broken test."""
    if job.get("kind") != "test":
        return "infra"
    try:
        return "test" if get_failures(str(job["id"])).get("failures") else "infra"
    except Exception:
        return "test"  # if we truly can't tell, don't cry wolf about infra


def render_platform(run_id, sha, platform, refresh=False):
    try:
        jobs = get_jobs(run_id, refresh=refresh)
    except Exception as ex:
        return f'<div class="muted">Could not load jobs: {e(ex)}</div>'
    if not jobs:
        return '<div class="muted">No jobs.</div>'

    # Split test failures from infra/runner failures for the failing cells — one
    # cached annotation call each, fanned out so a platform with many reds stays snappy.
    failing = [
        j
        for j in jobs
        if classify(j["status"], j["conclusion"])[0] == "fail"
        and j["kind"] in ("test", "build")
    ]
    fk = {}
    if failing:
        with ThreadPoolExecutor(max_workers=8) as ex:
            for j, k in zip(failing, ex.map(fail_kind, failing)):
                fk[j["id"]] = k

    # group by (kind, group)
    groups = {}
    for j in jobs:
        groups.setdefault((KIND_ORDER.get(j["kind"], 9), j["group"]), []).append(j)

    out = []
    for (_, gname), gjobs in sorted(
        groups.items(), key=lambda kv: (kv[0][0], kv[0][1])
    ):
        info = info_for(gname)
        tierpath = (
            f'<span class="tierpath">{e(", ".join(info["paths"]))}</span>'
            if info
            else ""
        )
        cells, rows = [], []
        for j in sorted(gjobs, key=lambda j: (j["python"] or "", j["cuda"] or "")):
            cls, label = classify(j["status"], j["conclusion"])
            failable = cls == "fail" and j["kind"] in ("test", "build")
            infra = failable and fk.get(j["id"]) == "infra"
            if infra:
                label = "infra"
            fcls = cls + (
                " infra" if infra else ""
            )  # add .infra shading when a runner (not a test) died
            if j["python"] and j["cuda"]:
                inner = (
                    f'<span class="k"><span class="dot {cls}"></span>{e(j["python"])}·{e(j["cuda"])}</span>'
                    f'<span class="v">{e(label)}</span>'
                )
                if failable:
                    q = qp(
                        job=j["id"],
                        sha=sha,
                        platform=platform,
                        tier=gname,
                        py=j["python"],
                        cuda=j["cuda"],
                        url=j["url"],
                    )
                    cells.append(
                        f'<div class="cell {fcls}" tabindex="0" onclick="openDrawer()" '
                        f'hx-get="/ui/failures?{q}" hx-target="#drawer-body" hx-swap="innerHTML">{inner}</div>'
                    )
                else:
                    cells.append(f'<div class="cell {fcls}">{inner}</div>')
            else:
                name = j["raw"].split(" / ")[-1] or j["group"]
                link = (
                    f' <a href="{e(j["url"])}" target="_blank" rel="noopener">log ↗</a>'
                    if j["url"]
                    else ""
                )
                extra = ""
                if failable:
                    q = qp(
                        job=j["id"],
                        sha=sha,
                        platform=platform,
                        tier=gname,
                        url=j["url"],
                    )
                    extra = (
                        f' · <a href="#" onclick="openDrawer();return false" '
                        f'hx-get="/ui/failures?{q}" hx-target="#drawer-body" hx-swap="innerHTML">details</a>'
                    )
                rows.append(
                    f'<div class="jobrow {fcls}"><span class="dot {cls}"></span>'
                    f'<span class="name">{e(name)}</span>'
                    f'<span class="jobstate {fcls}">{e(label)}</span>{link}{extra}</div>'
                )
        body = ""
        if cells:
            body += f'<div class="matrix">{"".join(cells)}</div>'
        if rows:
            body += f'<div class="rows">{"".join(rows)}</div>'
        out.append(f'<div class="jobgroup"><h3>{e(gname)} {tierpath}</h3>{body}</div>')
    return "".join(out)


def render_failures(job_id, sha, platform, tier, py, cuda, url):
    data = get_failures(job_id)
    joblink = (
        (f' · <a href="{e(url)}" target="_blank" rel="noopener">job log ↗</a>')
        if url
        else ""
    )
    oob = (
        f'<span id="drawer-title" hx-swap-oob="true">{e(platform)} — {e(tier)}</span>'
        f'<span id="drawer-sub" hx-swap-oob="true">{e(py)} {e(cuda)}{joblink}</span>'
    )
    if data.get("error"):
        return (
            oob
            + f'<div class="muted">Could not load annotations: {e(data["error"])}</div>'
        )
    fails = data["failures"]
    info = info_for(tier)
    body = []
    if not fails:
        loglink = (
            f'<a href="{e(url)}" target="_blank" rel="noopener">Open the job log ↗</a>'
            if url
            else ""
        )
        body.append(
            '<div class="muted">No pytest failure annotations — this is likely a '
            f"build / environment / setup failure rather than a test assertion.<br><br>{loglink}</div>"
        )
    for f in fails:
        loc = ""
        if f["file"]:
            gh_url = (
                f'https://github.com/{repo_slug()}/blob/{sha}/{f["file"]}#L{f["line"]}'
            )
            loc = (
                f'<div class="floc">↳ <a href="{e(gh_url)}" target="_blank" rel="noopener">'
                f'<code>{e(f["file"])}:{f["line"]}</code> ↗</a></div>'
            )
        cat = categorize_failure(f["message"])
        body.append(
            f'<div class="fail-item"><div class="ftest">{e(f["test"] or "failure")}'
            f" {_cat_tag(cat)}</div>"
            f'{loc}<pre>{e(f["message"])}</pre></div>'
        )
    if info:
        leaves = []
        for f in fails:
            if f["test"]:
                leaf = re.split(r"[.:]", f["test"])[-1]
                leaf = re.sub(r"\[.*$", "", leaf)
                if leaf and leaf not in leaves:
                    leaves.append(leaf)
        kexpr = " or ".join(leaves[:6])
        cmd = info["repro"](kexpr)
        body.append(
            f'<div class="repro"><button class="copy" onclick="copyPre(this)">copy</button>'
            f'<div class="lbl">reproduce locally</div><code>{e(cmd)}</code></div>'
        )
    # Lazy: fetch the job log + extract the relevant block(s) once the drawer is in.
    body.append(
        f'<div class="logsec" hx-get="/ui/joblog?{qp(job=job_id, url=url)}" '
        f'hx-trigger="load" hx-swap="innerHTML">'
        f'<div class="muted" style="padding:8px 0"><span class="spinner"></span> '
        f"fetching relevant logs…</div></div>"
    )
    return oob + "".join(body)


def render_aggregate(branch, refresh=False):
    """Cross-platform failure rollup: dedupe a failing test across every platform
    it breaks on, so 'fails on N platforms' points straight at the likely code."""
    try:
        runs = get_runs(branch, refresh=refresh)
    except Exception as ex:
        return f'<div class="muted">Could not scan: {e(ex)}</div>'

    def jobs_of(r):
        try:
            return [(r, j) for j in get_jobs(r["id"], refresh=refresh)]
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=8) as ex:
        alljobs = [x for sub in ex.map(jobs_of, runs) for x in sub]
    failed = [
        (r, j)
        for r, j in alljobs
        if classify(j["status"], j["conclusion"])[0] == "fail" and j["kind"] == "test"
    ]
    if not failed:
        healthy = all(classify(r["status"], r["conclusion"])[0] != "fail" for r in runs)
        msg = (
            "No failing test jobs 🎉"
            if healthy
            else "No test-level failures found (failures are build/setup — see the platform grids)."
        )
        return f'<div class="agg-ok">{msg}</div>'

    def fails_of(rj):
        r, j = rj
        return (r, j, get_failures(j["id"], refresh=refresh).get("failures", []))

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(fails_of, failed))

    agg = {}
    for r, j, fails in results:
        plat = pretty_platform(r["workflowName"])
        if not fails:  # failed job but no parseable test → bucket under the tier
            key = f"[{j['group']}]"
            a = agg.setdefault(key, dict(test=key, file=None, line=None, occ=[]))
            a["occ"].append(
                dict(plat=plat, py=j["python"], cuda=j["cuda"], url=j["url"], msg="")
            )
            continue
        for f in fails:
            key = f["test"] or f["message"].splitlines()[0]
            a = agg.setdefault(
                key, dict(test=f["test"] or key, file=f["file"], line=f["line"], occ=[])
            )
            a["file"] = a["file"] or f["file"]
            a["line"] = a["line"] or f["line"]
            a["occ"].append(
                dict(
                    plat=plat,
                    py=j["python"],
                    cuda=j["cuda"],
                    url=j["url"],
                    msg=f["message"],
                )
            )

    # Config universe (what actually ran) so "X only" hints are real subsets.
    test_jobs = [j for _, j in alljobs if j["kind"] == "test"]
    all_pys = {j["python"] for j in test_jobs if j["python"]}
    all_cudas = {j["cuda"] for j in test_jobs if j["cuda"]}

    # Categorize once per row (from the sample traceback) so we can tally + tag.
    for a in agg.values():
        a["sample"] = next((o["msg"] for o in a["occ"] if o["msg"]), "")
        a["cat"] = categorize_failure(a["sample"] or a["test"])

    # Real regressions (bug) first, then env/gap, infra last; ties → most cells.
    _KIND_RANK = {"bug": 0, "env": 1, "unknown": 2, "infra": 3}
    rows = sorted(
        agg.values(),
        key=lambda a: (
            _KIND_RANK.get(a["cat"]["kind"], 2),
            -len({o["plat"] for o in a["occ"]}),
            -len(a["occ"]),
            a["test"],
        ),
    )
    out = []
    for a in rows:
        plats = sorted({o["plat"] for o in a["occ"]})
        ncells = len(a["occ"])
        loc = ""
        if a["file"]:
            gh_url = f'https://github.com/{repo_slug()}/blob/{runs[0].get("headSha","")}/{a["file"]}#L{a["line"]}'
            loc = f'<div class="agg-file">↳ <a href="{e(gh_url)}" target="_blank" rel="noopener"><code>{e(a["file"])}:{a["line"]}</code> ↗</a></div>'
        chips = "".join(f'<span class="chip">{e(p)}</span>' for p in plats)
        pat = _config_pattern(a["occ"], all_pys, all_cudas)
        patchips = "".join(f'<span class="cfgpat">{e(p)}</span>' for p in pat)
        out.append(f"""<details class="agg-row-d">
      <summary class="agg-row">
        <div><div class="agg-test">{e(a["test"])} {_cat_tag(a["cat"])}{patchips}</div>{loc}</div>
        <span class="agg-count"><span class="dot fail"></span>{len(plats)} platform{"s" if len(plats)!=1 else ""} · {ncells} cell{"s" if ncells!=1 else ""}</span>
      </summary>
      <div class="agg-detail"><div class="chips">{chips}</div>{f"<pre>{e(a['sample'])}</pre>" if a["sample"] else ""}</div>
    </details>""")
    # Category tally — an instant read on the run's character (infra flakes vs regressions).
    tally = {}
    for a in rows:
        tally.setdefault(a["cat"]["kind"], dict(n=0, label=a["cat"]["label"]))["n"] += 1
    order = ["bug", "env", "unknown", "infra"]
    tstr = " · ".join(
        f'<span class="cat {k}">{tally[k]["n"]} {e(_KIND_LABEL[k])}</span>'
        for k in order
        if k in tally
    )
    header = (
        f'<div class="agg-head-row"><span>{len(rows)} distinct failing test'
        f'{"s" if len(rows)!=1 else ""} across '
        f'{len({o["plat"] for a in rows for o in a["occ"]})} platforms</span>'
        f'<span class="agg-tally">{tstr}</span>'
        f'<button class="btn small" hx-get="/ui/aggregate?branch={e(branch)}&refresh=1" '
        f'hx-target="#agg" hx-swap="innerHTML">↻ rescan</button></div>'
    )
    return header + f'<div class="agg">{"".join(out)}</div>'


# ── HTTP server ──────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body, ctype = json.dumps(body), "application/json"
        body = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # Never let the browser serve a stale page/fragment/asset — this is a
        # live, rapidly-changing tool and cached HTML/CSS/JS causes "it broke
        # again" ghosts. gh data is already cached server-side (Cache class).
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        refresh = q.get("refresh") == "1"
        try:
            p = u.path
            if p in ("/", "/index.html"):
                return self._index(q.get("branch") or default_branch())
            if p.startswith("/static/"):
                return self._static(p[len("/static/") :])
            if p == "/ui/board":
                return self._send(
                    200, render_board(q.get("branch") or default_branch(), refresh)
                )
            if p == "/ui/status":
                return self._send(
                    200, render_status(q.get("branch") or default_branch())
                )
            if p == "/ui/platform":
                return self._send(
                    200,
                    render_platform(
                        q["run"], q.get("sha", ""), q.get("platform", ""), refresh
                    ),
                )
            if p == "/ui/failures":
                return self._send(
                    200,
                    render_failures(
                        q["job"],
                        q.get("sha", ""),
                        q.get("platform", ""),
                        q.get("tier", ""),
                        q.get("py", ""),
                        q.get("cuda", ""),
                        q.get("url", ""),
                    ),
                )
            if p == "/api/resolve-pr":
                b = resolve_pr_branch(q.get("pr", ""))
                return self._send(200, {"branch": b} if b else {"error": "not found"})
            if p == "/ui/aggregate":
                return self._send(
                    200, render_aggregate(q.get("branch") or default_branch(), refresh)
                )
            if p == "/ui/joblog":
                if q.get("raw") == "1":
                    log = get_job_log(q["job"], refresh)
                    return self._send(
                        200 if log else 404,
                        log or "log not available",
                        "text/plain; charset=utf-8",
                    )
                return self._send(200, render_joblog(q["job"], q.get("url", "")))
            return self._send(404, "not found", "text/plain")
        except KeyError as ex:
            self._send(400, f"missing param {ex}", "text/plain")
        except Exception as ex:
            self._send(500, f'<div class="muted">error: {e(ex)}</div>')

    def do_POST(self):
        u = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            form = {
                k: v[0] for k, v in parse_qs(self.rfile.read(length).decode()).items()
            }
            if u.path == "/api/pr-comment":
                try:
                    url = post_pr_comment(form.get("pr", ""), form.get("cmd", ""))
                    return self._send(200, {"ok": True, "url": url})
                except Exception as ex:  # bad cmd / gh failure → readable msg
                    return self._send(200, {"ok": False, "error": str(ex)})
            if u.path == "/api/pr-label":
                try:
                    out = set_pr_label(
                        form.get("pr", ""),
                        form.get("label", ""),
                        form.get("add") == "1",
                    )
                    return self._send(200, {"ok": True, "out": out})
                except Exception as ex:  # bad label / gh failure → readable msg
                    return self._send(200, {"ok": False, "error": str(ex)})
            return self._send(404, {"ok": False, "error": "not found"})
        except Exception as ex:
            return self._send(500, {"ok": False, "error": str(ex)})

    def _index(self, branch):
        """Full page bootstrapped to `branch` — so `/?branch=<x>` is a real,
        reloadable, shareable URL (and history-restore lands on a valid page).
        A bare `#<num>` in the URL is a PR number → resolve to its head branch."""
        if branch and branch.startswith("#") and branch[1:].isdigit():
            branch = resolve_pr_branch(branch) or branch
        with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as f:
            html_s = f.read()
        html_s = html_s.replace("{{BRANCH}}", e(branch)).replace(
            "{{REPO}}", e(repo_slug())
        )
        self._send(200, html_s)

    def _static(self, rel):
        rel = rel.split("?")[0].lstrip("/")
        path = os.path.normpath(os.path.join(STATIC, rel))
        if not path.startswith(STATIC) or not os.path.isfile(path):
            return self._send(404, "not found", "text/plain")
        ctype = {"html": "text/html", "js": "text/javascript", "css": "text/css"}.get(
            rel.rsplit(".", 1)[-1], "application/octet-stream"
        )
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype + "; charset=utf-8")


def preflight():
    try:
        subprocess.run(["gh", "auth", "status"], capture_output=True, check=True)
    except Exception:
        sys.exit("error: `gh` is not authenticated. Run `gh auth login` first.")


def tailscale_ip():
    """Best-effort Tailscale IPv4 (100.64.0.0/10). Uses the CLI if present,
    otherwise sniffs local interfaces. Returns None if not on a tailnet."""
    try:
        out = (
            subprocess.run(
                ["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=3
            )
            .stdout.strip()
            .splitlines()
        )
        if out and out[0]:
            return out[0].strip()
    except Exception:
        pass
    try:
        for res in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = res[4][0]
            if ip.startswith("100.") and 64 <= int(ip.split(".")[1]) <= 127:
                return ip
    except Exception:
        pass
    return None


def lan_ip():
    """Primary outbound-facing LAN IP (no packets sent)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def main():
    global _DEFAULT_BRANCH
    ap = argparse.ArgumentParser(description="Local Torch-TensorRT CI dashboard")
    ap.add_argument("-b", "--branch", default=None, help="default branch to show")
    ap.add_argument("-p", "--port", type=int, default=8712)
    ap.add_argument(
        "--host",
        default="0.0.0.0",
        help="bind address (default 0.0.0.0 — reachable over LAN/Tailscale; "
        "use 127.0.0.1 to keep it local-only)",
    )
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    preflight()
    if args.branch:
        _DEFAULT_BRANCH = args.branch
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    local = f"http://127.0.0.1:{args.port}/"
    print(f"CI dashboard  (repo {repo_slug()}, branch {default_branch()})")
    print(f"  local     {local}")
    if args.host in ("0.0.0.0", "::"):
        ts = tailscale_ip()
        if ts:
            print(f"  tailscale http://{ts}:{args.port}/")
        lan = lan_ip()
        if lan and lan != ts:
            print(f"  lan       http://{lan}:{args.port}/")
        print(
            "  (bound to all interfaces — anyone who can reach this host can view CI status)"
        )
    print("Ctrl-C to stop.", flush=True)
    if not args.no_open:
        try:
            import webbrowser

            webbrowser.open(local)  # always open the loopback URL locally
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
