"""
Pytest configuration for annotation tests.

GPU routing
-----------
Tests marked ``@pytest.mark.requires_pre_bw`` need a pre-Blackwell GPU
(compute capability < 10).  On Blackwell/Myelin, TRT's multi-tactic plugin
handling produces incorrect results (same root cause as the SymIntExprs
length-mismatch bug filed against TRT 10.14).

When the full annotation suite is run without ``CUDA_VISIBLE_DEVICES``:

  1. Main process  → Blackwell GPU (if available).  ``requires_pre_bw``
     tests are *skipped* in this pass.
  2. ``pytest_sessionfinish`` reruns all ``requires_pre_bw`` test files
     in a fresh subprocess with ``CUDA_VISIBLE_DEVICES`` pointing at
     the first pre-Blackwell GPU found via nvidia-smi.

CI / direct invocation
-----------------------
To run only pre-Blackwell tests::

    CUDA_VISIBLE_DEVICES=<a100-uuid> pytest -m requires_pre_bw tests/py/annotation/

To run only Blackwell tests::

    CUDA_VISIBLE_DEVICES=<bw-uuid> pytest -m "not requires_pre_bw" tests/py/annotation/
"""

import os
import subprocess
import sys

import pytest

_repo_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
_py_dir = os.path.join(_repo_root, "py")
if os.path.isdir(_py_dir) and _py_dir not in sys.path:
    sys.path.insert(0, _py_dir)

def _ensure_cublas_on_ld_path():
    for p in sys.path:
        if "site-packages" not in str(p):
            continue
        for root, _, files in os.walk(p):
            for f in files:
                if f.startswith("libcublas") and ".so" in f:
                    lp = os.environ.get("LD_LIBRARY_PATH", "")
                    if root not in lp:
                        os.environ["LD_LIBRARY_PATH"] = root + (":" + lp if lp else "")
                    return
    for base in (
        "/root/.pyenv/versions/*/lib/python*/site-packages/nvidia/cu*/lib",
        "/usr/local/lib/python*/site-packages/nvidia/cu*/lib",
    ):
        import glob
        for path in glob.glob(base):
            if os.path.isdir(path) and (
                os.path.isfile(os.path.join(path, "libcublas.so.13"))
                or os.path.isfile(os.path.join(path, "libcublas.so.12"))
                or os.path.isfile(os.path.join(path, "libcublas.so"))
            ):
                lp = os.environ.get("LD_LIBRARY_PATH", "")
                if path not in lp:
                    os.environ["LD_LIBRARY_PATH"] = path + (":" + lp if lp else "")
                return


def _gpu_map_nvsmi():
    """Return (blackwell_uuids, pre_blackwell_uuids) via nvidia-smi."""
    bw_uuids, pre_uuids = [], []
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=compute_cap,uuid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in out.strip().splitlines():
            parts = line.split(",")
            if len(parts) < 2:
                continue
            cc_major = int(parts[0].strip().split(".")[0])
            uuid = parts[1].strip()
            if cc_major >= 10:
                bw_uuids.append(uuid)
            else:
                pre_uuids.append(uuid)
    except Exception:
        pass
    return bw_uuids, pre_uuids


_IS_PRE_BW_SUBPROCESS = os.environ.get("_TTA_PRE_BW_SUBPROCESS") == "1"

_BW_GPUS, _PRE_BW_GPUS = _gpu_map_nvsmi()

_ensure_cublas_on_ld_path()

collect_ignore: list = []
if _IS_PRE_BW_SUBPROCESS:
    # Pin to a pre-Blackwell GPU if the caller did not already set one.
    if "CUDA_VISIBLE_DEVICES" not in os.environ and _PRE_BW_GPUS:
        os.environ["CUDA_VISIBLE_DEVICES"] = _PRE_BW_GPUS[0]
elif _BW_GPUS:
    # Main run on Blackwell: pin to Blackwell GPUs.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(_BW_GPUS)


# ---------------------------------------------------------------------------
# Mark-based GPU routing
# ---------------------------------------------------------------------------

def pytest_runtest_setup(item):
    """Skip requires_pre_bw tests when running on Blackwell (main pass).

    They will be re-executed on a pre-Blackwell GPU in pytest_sessionfinish.
    """
    if _IS_PRE_BW_SUBPROCESS:
        return
    if not _BW_GPUS:
        return  # no Blackwell GPU — run everything here
    if item.get_closest_marker("requires_pre_bw"):
        pytest.skip(
            "requires_pre_bw: skipped on Blackwell; "
            "runs automatically in a pre-Blackwell subprocess after the main pass. "
            "To run directly: CUDA_VISIBLE_DEVICES=<pre-bw-uuid> pytest -m requires_pre_bw"
        )


@pytest.fixture(autouse=True)
def clear_aot_caches():
    """No-op fixture retained for API compatibility."""
    yield


# ---------------------------------------------------------------------------
# Per-worker timing cache — speeds up repeated TRT builds within one session
# ---------------------------------------------------------------------------

# Persistent per-worker timing cache directory. Using ~/.cache (not /tmp)
# so the cache survives reboots: the first run pays the full tactic-selection
# cost; subsequent runs reuse previously measured tactics and are significantly
# faster (especially for INT8/FP8 builds).
_TC_DIR: str = os.path.join(os.path.expanduser("~"), ".cache", "tta_test_timing_cache")
os.makedirs(_TC_DIR, exist_ok=True)


@pytest.fixture(autouse=True, scope="session")
def _session_timing_cache():
    """Inject a per-worker TRT timing cache into every ``torch_tensorrt.compile`` call.

    With ``editable_timing_cache=True``, TRT writes each newly selected tactic
    to the cache file so subsequent builds of the same op+shape+dtype skip
    tactic selection entirely.

    Each worker process uses its own cache file (keyed by ``PYTEST_XDIST_WORKER``
    id) to avoid concurrent-write corruption under ``-n 12``.  The cache dir is
    **not** deleted at session end so the entries accumulate across runs.
    """
    import torch_tensorrt.dynamo

    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")
    cache_path = os.path.join(_TC_DIR, f"tc_{worker_id}.bin")

    _orig_compile = torch_tensorrt.dynamo.compile

    def _patched_compile(*args, **kwargs):
        kwargs.setdefault("timing_cache_path", cache_path)
        kwargs.setdefault("editable_timing_cache", True)
        return _orig_compile(*args, **kwargs)

    torch_tensorrt.dynamo.compile = _patched_compile
    yield
    torch_tensorrt.dynamo.compile = _orig_compile


def pytest_sessionfinish(session, exitstatus):
    if _IS_PRE_BW_SUBPROCESS or not _PRE_BW_GPUS:
        return

    # --- requires_pre_bw tests ---
    # Collect the unique test-file paths for all requires_pre_bw items.
    pre_bw_files: list = []
    seen: set = set()
    for item in session.items:
        if item.get_closest_marker("requires_pre_bw"):
            path = str(item.fspath)
            if path not in seen:
                seen.add(path)
                pre_bw_files.append(path)

    if pre_bw_files:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = _PRE_BW_GPUS[0]
        env["_TTA_PRE_BW_SUBPROCESS"] = "1"
        n_workers = str(session.config.getoption("numprocesses", default=4))
        cmd = (
            [sys.executable, "-m", "pytest", "-v", "--tb=short", "-n", n_workers,
             "-m", "requires_pre_bw"]
            + pre_bw_files
        )
        result = subprocess.run(cmd, env=env, cwd=str(session.config.rootpath))
        if result.returncode != 0 and exitstatus == 0:
            session.exitstatus = result.returncode
