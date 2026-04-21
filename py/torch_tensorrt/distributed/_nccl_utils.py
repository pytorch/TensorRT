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

import ctypes
import logging
import os
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


def check_nccl_library_path() -> bool:
    """
    Check if LD_LIBRARY_PATH includes PyTorch's NCCL directory.

    Returns:
        True if configuration is correct, False if LD_LIBRARY_PATH needs updating.
    """
    nccl_lib_dir = get_nccl_library_path()

    if nccl_lib_dir is None:
        # System NCCL - no action needed
        return True

    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    return nccl_lib_dir in ld_library_path


def setup_nccl_for_torch_tensorrt() -> None:
    """
    Setup NCCL library for TensorRT distributed inference.

    This function:
    1. Detects if nvidia.nccl pip package is installed
    2. Creates libnccl.so symlink if needed
    3. Pre-loads libnccl.so via ctypes (helps Python runtime path)
    4. Updates LD_LIBRARY_PATH for dynamic loaders

    Note: TRT's internal loader (libLoader.cpp) reads LD_LIBRARY_PATH at
    process launch time, not when updated via os.environ. For the C++ TRT
    runtime path, LD_LIBRARY_PATH must be set before the process starts:

        NCCL_LIB=$(python -c "from torch_tensorrt.distributed._nccl_utils import get_nccl_library_path; print(get_nccl_library_path())")
        LD_LIBRARY_PATH="$NCCL_LIB:$LD_LIBRARY_PATH" python script.py

    For NGC containers (system NCCL), this is a no-op.
    """
    global _nccl_setup_checked

    # Only check once per process
    if _nccl_setup_checked:
        return
    _nccl_setup_checked = True

    nccl_lib_dir = get_nccl_library_path()

    if nccl_lib_dir is None:
        # NGC container or system NCCL - no action needed
        logger.debug(
            "nvidia.nccl package not found. "
            "Assuming system NCCL is used by both PyTorch and TensorRT."
        )
        return

    logger.debug(f"Found nvidia.nccl package at: {nccl_lib_dir}")

    # Ensure symlink exists
    symlink_ok = ensure_nccl_symlink(nccl_lib_dir)

    # Ensure LD_LIBRARY_PATH includes the NCCL directory so TRT's dlopen("libnccl.so")
    # finds the same library PyTorch already loaded.  dlopen() reads LD_LIBRARY_PATH
    # dynamically, so updating os.environ here takes effect for subsequent loads.
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if nccl_lib_dir not in ld_library_path:
        os.environ["LD_LIBRARY_PATH"] = (
            f"{nccl_lib_dir}:{ld_library_path}" if ld_library_path else nccl_lib_dir
        )
        logger.debug(f"Added NCCL directory to LD_LIBRARY_PATH: {nccl_lib_dir}")
    else:
        logger.debug(f"LD_LIBRARY_PATH already includes NCCL directory: {nccl_lib_dir}")

    if symlink_ok:
        # Pre-load libnccl.so into the process with RTLD_GLOBAL so that TRT's
        # subsequent dlopen("libnccl.so") inside setCommunicator() finds the
        # already-loaded library rather than searching LD_LIBRARY_PATH again.
        nccl_so = os.path.join(nccl_lib_dir, "libnccl.so")
        try:
            ctypes.CDLL(nccl_so, mode=ctypes.RTLD_GLOBAL)
            logger.debug(f"Pre-loaded NCCL library: {nccl_so}")
        except OSError as e:
            logger.warning(f"Failed to pre-load NCCL library {nccl_so}: {e}")

        logger.debug("NCCL library setup complete")


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
    """Warn if an requires_native_multidevice TRT engine's NCCL prerequisites are not satisfied.

    Checks two conditions and logs a warning for each:
    1. LD_LIBRARY_PATH does not include PyTorch's NCCL lib dir (too late to fix,
       must be set before process launch — use torchtrtrun).
    2. torch.distributed is not initialized or world_size == 1.

    Call this from both TorchTensorRTModule and PythonTorchTensorRTModule after
    confirming the engine has NCCL collective ops.
    """
    if get_nccl_library_path() is not None and not check_nccl_library_path():
        logger.warning(
            "This TRT engine contains NCCL collective ops but "
            "LD_LIBRARY_PATH does not include PyTorch's NCCL library directory. "
            "TRT may load a different NCCL instance than PyTorch, causing "
            "communicator sharing to fail. Use torchtrtrun to launch distributed "
            "scripts, or set LD_PRELOAD and LD_LIBRARY_PATH before process start:\n"
            "  NCCL_LIB=$(python -c 'from torch_tensorrt.distributed._nccl_utils "
            "import get_nccl_library_path; print(get_nccl_library_path())')\n"
            "  LD_PRELOAD=$NCCL_LIB/libnccl.so.2 "
            "LD_LIBRARY_PATH=$NCCL_LIB:$LD_LIBRARY_PATH python ..."
        )

    if not (
        dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    ):
        logger.warning(
            "This TRT engine contains NCCL collective ops but torch.distributed "
            "is not initialized (or world_size=1). Call "
            "dist.init_process_group(backend='nccl') before running this engine, "
            "otherwise results will be incorrect."
        )
