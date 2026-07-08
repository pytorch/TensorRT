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
    core / L2 dynamo distributed tests / L2-dynamo-distributed-tests--3.12-cu130
We parse that into {group/tier, python, cuda, kind} and map the tier back to the
pytest paths in tests/py/utils/ci_helpers.sh so a red cell points at code.
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

# ── Tier → source mapping ────────────────────────────────────────────────────
# Mirrors the trt_tier_* selectors in tests/py/utils/ci_helpers.sh (the single
# source of truth for "what does each CI tier run"). Keys are normalized job
# group names (see _norm()); `paths` are repo-relative; `fn` is the shell tier
# function you'd invoke locally to reproduce.
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


def tier_for(group: str):
    return TIER_MAP.get(_norm(group))


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
    if "build-wheel" in last or gl.startswith("build "):
        kind = "build"
    elif re.match(r"l[012]\b", gl):
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
TESTLINE = re.compile(r"^(?P<test>[\w./-]+(?:\.[\w\[\]-]+)+)\s*$")


@lru_cache(maxsize=4096)
def grep_test(symbol):
    """Resolve a failing-test symbol (Class.method / module.func) to (file, line)
    via `git grep`. Returns None if not found."""
    leaf = re.split(r"[.:]", symbol.strip())[-1]
    leaf = re.sub(r"\[.*$", "", leaf)
    if not re.match(r"^[A-Za-z_]\w+$", leaf):
        return None
    pat = f"def {leaf}\\b" if leaf.startswith("test") else f"class {leaf}\\b"
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
        m = TESTLINE.match(head)
        test = m.group("test") if m else None
        dedupe = test or head
        if dedupe in seen:
            continue
        seen.add(dedupe)
        loc = grep_test(test) if test else None
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
    return [_TS.sub("", ln) for ln in raw.splitlines()]


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


def summary_reasons(lines, test):
    """The concise `FAILED …/ERROR … - <reason>` lines from pytest's summary."""
    leaf = re.split(r"[.:]", test)[-1].split("[")[0] if test else None
    return [
        ln
        for ln in lines
        if re.match(r"^(FAILED|ERROR) ", ln) and (not leaf or leaf in ln)
    ]


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
    fails = get_failures(job_id).get("failures", [])
    blocks = []
    for f in fails:
        excerpt = extract_test_log(lines, f["test"])
        reasons = summary_reasons(lines, f["test"])
        title = e(f["test"] or "failure")
        reason_html = (
            f'<div class="logreason">{e(reasons[0][:300])}</div>' if reasons else ""
        )
        if excerpt:
            blocks.append(
                f'<details class="logblock" open><summary>{title}</summary>'
                f"{reason_html}<pre>{e(excerpt)}</pre></details>"
            )
        else:
            blocks.append(
                f'<details class="logblock"><summary>{title}</summary>'
                f'{reason_html}<div class="muted">No matching block in the log — '
                f"see the raw log.</div></details>"
            )
    if not fails:  # build/env/setup failure: no test annotations → show the tail
        tail = "\n".join(lines[-180:]).strip()
        blocks.append(
            '<details class="logblock" open><summary>end of job log</summary>'
            f"<pre>{e(tail)}</pre></details>"
        )
    return head + "".join(blocks)


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
    sha = (head.get("headSha") or "")[:9]
    commit = (
        f'<div class="commit"><code>{e(sha)}</code> · {e(head.get("displayTitle", ""))[:80]}'
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
    return summary + agg + board + poller


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


def render_platform(run_id, sha, platform, refresh=False):
    try:
        jobs = get_jobs(run_id, refresh=refresh)
    except Exception as ex:
        return f'<div class="muted">Could not load jobs: {e(ex)}</div>'
    if not jobs:
        return '<div class="muted">No jobs.</div>'

    # group by (kind, group)
    groups = {}
    for j in jobs:
        groups.setdefault((KIND_ORDER.get(j["kind"], 9), j["group"]), []).append(j)

    out = []
    for (_, gname), gjobs in sorted(
        groups.items(), key=lambda kv: (kv[0][0], kv[0][1])
    ):
        tier = tier_for(gname)
        tierpath = (
            f'<span class="tierpath">{e(", ".join(tier["paths"]))}</span>'
            if tier
            else ""
        )
        cells, rows = [], []
        for j in sorted(gjobs, key=lambda j: (j["python"] or "", j["cuda"] or "")):
            cls, label = classify(j["status"], j["conclusion"])
            failable = cls == "fail" and j["kind"] in ("test", "build")
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
                        f'<div class="cell fail" tabindex="0" onclick="openDrawer()" '
                        f'hx-get="/ui/failures?{q}" hx-target="#drawer-body" hx-swap="innerHTML">{inner}</div>'
                    )
                else:
                    cells.append(f'<div class="cell {cls}">{inner}</div>')
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
                    f'<div class="jobrow {cls}"><span class="dot {cls}"></span>'
                    f'<span class="name">{e(name)}</span>'
                    f'<span class="jobstate {cls}">{e(label)}</span>{link}{extra}</div>'
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
    tinfo = tier_for(tier)
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
        body.append(
            f'<div class="fail-item"><div class="ftest">{e(f["test"] or "failure")}</div>'
            f'{loc}<pre>{e(f["message"])}</pre></div>'
        )
    if tinfo:
        leaves = []
        for f in fails:
            if f["test"]:
                leaf = re.split(r"[.:]", f["test"])[-1]
                leaf = re.sub(r"\[.*$", "", leaf)
                if leaf and leaf not in leaves:
                    leaves.append(leaf)
        k = f' -k "{" or ".join(leaves[:6])}"' if leaves else ""
        cmd = (
            f"source tests/py/utils/ci_helpers.sh && "
            f'PYTHON="uv run --no-sync python" TRT_PYTEST_RERUNS=0 {tinfo["fn"]}{k}'
        )
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

    rows = sorted(
        agg.values(),
        key=lambda a: (-len({o["plat"] for o in a["occ"]}), -len(a["occ"]), a["test"]),
    )
    out = []
    for a in rows:
        plats = sorted({o["plat"] for o in a["occ"]})
        loc = ""
        if a["file"]:
            gh_url = f'https://github.com/{repo_slug()}/blob/{runs[0].get("headSha","")}/{a["file"]}#L{a["line"]}'
            loc = f'<div class="agg-file">↳ <a href="{e(gh_url)}" target="_blank" rel="noopener"><code>{e(a["file"])}:{a["line"]}</code> ↗</a></div>'
        chips = "".join(f'<span class="chip">{e(p)}</span>' for p in plats)
        sample = next((o["msg"] for o in a["occ"] if o["msg"]), "")
        out.append(f"""<details class="agg-row-d">
      <summary class="agg-row">
        <div><div class="agg-test">{e(a["test"])}</div>{loc}</div>
        <span class="agg-count"><span class="dot fail"></span>{len(plats)} platform{"s" if len(plats)!=1 else ""}</span>
      </summary>
      <div class="agg-detail"><div class="chips">{chips}</div>{f"<pre>{e(sample)}</pre>" if sample else ""}</div>
    </details>""")
    header = (
        f'<div class="agg-head-row">{len(rows)} distinct failing test'
        f'{"s" if len(rows)!=1 else ""} across {len({o["plat"] for a in rows for o in a["occ"]})} platforms'
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
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        refresh = q.get("refresh") == "1"
        try:
            p = u.path
            if p in ("/", "/index.html"):
                return self._index()
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

    def _index(self):
        with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as f:
            html_s = f.read()
        html_s = html_s.replace("{{BRANCH}}", e(default_branch())).replace(
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
