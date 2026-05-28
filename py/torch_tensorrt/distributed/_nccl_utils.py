"""
NCCL Library Utilities for Distributed TensorRT Inference

This module handles NCCL library path resolution to ensure TensorRT and PyTorch
use the same NCCL library instance. This is critical for sharing NCCL communicators
between PyTorch's distributed backend and TensorRT's native NCCL collectives.

Background:
-----------
TensorRT's dlopen("libnccl.so") may load a different NCCL library than PyTorch,
causing crashes when sharing NCCL communicators.

- PyTorch loads NCCL via RPATH baked at compile time (libnccl.so.2)
- TensorRT lazy-loads NCCL via dlopen("libnccl.so") at runtime

The mismatch occurs because:
1. pip's nvidia-nccl-cu* package only ships libnccl.so.2 (no libnccl.so symlink)
2. TRT specifically looks for libnccl.so, misses pip's copy, falls back to system NCCL

Environments:
-------------
- NGC containers: No action needed (both use system NCCL)
- pip install torch: Requires symlink + LD_LIBRARY_PATH setup

Future:
-------
TensorRT 11.0 will support TRT_NCCL_LIBRARY env var, eliminating the need for
symlink workarounds.
"""

import logging
import os
import subprocess
from typing import Optional

import torch.distributed as dist

logger = logging.getLogger(__name__)

_nccl_setup_checked = False


def get_nccl_library_path() -> Optional[str]:
    """
    Get the path to PyTorch's NCCL library directory.

    Returns:
        Path to NCCL lib directory if nvidia.nccl package exists, None otherwise.
        None indicates system NCCL is being used (e.g., NGC containers).
    """
    try:
        import nvidia.nccl

        nccl_lib_dir = os.path.join(list(nvidia.nccl.__path__)[0], "lib")
        if os.path.isdir(nccl_lib_dir):
            return nccl_lib_dir
        return None
    except ImportError:
        # nvidia.nccl not installed - using system NCCL (e.g., NGC container)
        return None


def ensure_nccl_symlink(nccl_lib_dir: str) -> bool:
    """
    Ensure libnccl.so symlink exists pointing to libnccl.so.2.

    TensorRT's dlopen looks for "libnccl.so", but pip's nvidia-nccl package
    only ships "libnccl.so.2". This creates the necessary symlink.

    Args:
        nccl_lib_dir: Path to the NCCL library directory

    Returns:
        True if symlink exists or was created, False otherwise.
    """
    nccl_so = os.path.join(nccl_lib_dir, "libnccl.so")
    nccl_so_2 = os.path.join(nccl_lib_dir, "libnccl.so.2")

    # Check if symlink already exists
    if os.path.lexists(nccl_so):
        return True

    # Check if target exists
    if not os.path.exists(nccl_so_2):
        logger.warning(f"NCCL library not found at {nccl_so_2}")
        return False

    # Try to create symlink
    try:
        os.symlink("libnccl.so.2", nccl_so)
        logger.info(f"Created NCCL symlink: {nccl_so} -> libnccl.so.2")
        return True
    except PermissionError:
        logger.warning(
            f"Cannot create NCCL symlink at {nccl_so} (permission denied). "
            f"Please run: ln -sf libnccl.so.2 {nccl_so}"
        )
        return False
    except OSError as e:
        logger.warning(f"Failed to create NCCL symlink: {e}")
        return False


def _sys_libdir_on_ldso_path() -> str:
    """Pick a system library directory that ld.so searches by default.

    Returns the first existing directory from a portability-ordered list:
    Debian/Ubuntu x86_64 multiarch → ARM64 multiarch → RHEL/CentOS lib64 →
    bare /usr/lib (always on ld.so's search path as a final fallback).
    """
    for d in (
        "/usr/lib/x86_64-linux-gnu",  # Debian / Ubuntu x86_64
        "/usr/lib/aarch64-linux-gnu",  # Debian / Ubuntu ARM64 (Jetson)
        "/usr/lib64",  # RHEL / CentOS / Fedora x86_64
    ):
        if os.path.isdir(d):
            return d
    return "/usr/lib"


def setup_nccl_for_torch_tensorrt() -> None:
    """
    Point a `libnccl.so` symlink on ld.so's default search path at PyTorch's
    libnccl.so.2 so TRT and PyTorch share a single NCCL library in the process.

    What this function does:
      1. Locate the nvidia.nccl pip package's libnccl.so.2 via
         get_nccl_library_path().  If pip's nccl isn't installed (NGC /
         system-NCCL environments) returns immediately — no action needed.
      2. Pick a system library directory that ld.so already searches by
         default via _sys_libdir_on_ldso_path() (Debian/Ubuntu multiarch,
         RHEL lib64, or /usr/lib fallback).
      3. If <sys_libdir>/libnccl.so already points at that libnccl.so.2,
         return.
      4. Otherwise, remove any existing libnccl.so at that path and create
         a fresh symlink:
             <sys_libdir>/libnccl.so → <pip>/libnccl.so.2
      5. Run `ldconfig` to refresh /etc/ld.so.cache.
      6. Guarded by a module-global flag so subsequent calls in the same
         process are a no-op.

    Requires write access to the chosen sys_libdir (root inside Docker is
    the common case).  On OSError the function raises RuntimeError with
    documented LD_PRELOAD / LD_LIBRARY_PATH workarounds for non-root setups.
    """
    global _nccl_setup_checked
    if _nccl_setup_checked:
        return
    _nccl_setup_checked = True

    nccl_lib_dir = get_nccl_library_path()
    if nccl_lib_dir is None:
        logger.debug(
            "nvidia.nccl package not found; assuming system NCCL is shared by PyTorch and TensorRT."
        )
        return

    nccl_so_2 = os.path.join(nccl_lib_dir, "libnccl.so.2")
    if not os.path.isfile(nccl_so_2):
        logger.warning(
            f"Expected {nccl_so_2} to exist but it doesn't; skipping NCCL setup."
        )
        return

    sys_libdir = _sys_libdir_on_ldso_path()
    target = os.path.join(sys_libdir, "libnccl.so")

    try:
        if os.path.lexists(target):
            existing = os.readlink(target) if os.path.islink(target) else None
            if existing == nccl_so_2:
                logger.debug(f"{target} already points at {nccl_so_2}; nothing to do.")
                return
            os.remove(target)
        os.symlink(nccl_so_2, target)
        subprocess.run(["ldconfig"], check=False)
        logger.info(
            f"NCCL: linked {target} -> {nccl_so_2} so TRT and PyTorch share one libnccl."
        )
    except OSError as e:
        raise RuntimeError(
            f"setup_nccl_for_torch_tensorrt(): cannot write {target} "
            f"(needed so TRT's dlopen('libnccl.so') resolves to PyTorch's libnccl.so.2). "
            f"Workarounds without root: relaunch python with "
            f"LD_PRELOAD={nccl_so_2} ; or pre-set "
            f"LD_LIBRARY_PATH={nccl_lib_dir}:$LD_LIBRARY_PATH before python starts "
            f"(and create a libnccl.so symlink in that dir first). "
            f"Original error: {e}"
        ) from e


def initialize_nccl_comm(device: Optional[int] = None) -> None:
    """Eagerly initialize PyTorch's NCCL communicator for a device.

    TRT's C++ runtime binds the NCCL communicator from PyTorch's
    ProcessGroupNCCL via ``bind_nccl_comm()``.  However, PyTorch creates
    this communicator lazily — only after the first NCCL collective.
    If a TRT engine with NCCL ops (``requires_native_multidevice=True``) is loaded and executed
    before any collective has run, ``bind_nccl_comm()`` finds a null
    communicator and the engine's all-reduce produces incorrect results.

    Call this function after ``dist.init_process_group(backend='nccl')``
    and before loading/running a distributed TRT engine to ensure the
    communicator is ready.

    Args:
        device: CUDA device ordinal.  Defaults to ``torch.cuda.current_device()``.

    Example::

        import torch.distributed as dist
        from torch_tensorrt.distributed._nccl_utils import (
            setup_nccl_for_torch_tensorrt,
            initialize_nccl_comm,
        )

        dist.init_process_group(backend="nccl")
        setup_nccl_for_torch_tensorrt()
        initialize_nccl_comm()          # NCCL comm now ready for TRT

        loaded = torch_tensorrt.load("engine_rank0.ep")
        output = loaded.module()(input_ids)
    """
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialized. "
            "Call dist.init_process_group(backend='nccl') first."
        )

    if device is None:
        device = torch.cuda.current_device()

    pg = dist.distributed_c10d._get_default_group()
    try:
        nccl_backend = pg._get_backend(torch.device(f"cuda:{device}"))
    except RuntimeError as e:
        raise RuntimeError(
            f"Could not get NCCL backend for cuda:{device}. "
            "Ensure dist.init_process_group was called with backend='nccl'. "
            f"Original error: {e}"
        ) from e

    if not hasattr(nccl_backend, "eager_connect_single_device"):
        raise RuntimeError(
            "ProcessGroupNCCL.eager_connect_single_device is not available. "
            "This requires PyTorch 2.4+. As a workaround, run a dummy "
            "collective before loading the TRT engine:\n"
            "    _dummy = torch.zeros(1, device='cuda')\n"
            "    dist.all_reduce(_dummy)"
        )

    nccl_backend.eager_connect_single_device(torch.device(f"cuda:{device}"))
    logger.debug(f"NCCL communicator eagerly initialized for cuda:{device}")


def check_nccl_engine_requirements() -> None:
    """Warn if a requires_native_multidevice TRT engine's NCCL prerequisites are not satisfied.

    Called from TorchTensorRTModule and PythonTorchTensorRTModule after
    confirming the engine has NCCL collective ops.
    """
    if not (
        dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    ):
        logger.warning(
            "This TRT engine contains NCCL collective ops but torch.distributed "
            "is not initialized (or world_size=1). Call "
            "dist.init_process_group(backend='nccl') before running this engine, "
            "otherwise results will be incorrect."
        )
