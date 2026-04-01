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
from typing import Optional

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


def setup_nccl_library() -> None:
    """
    Setup NCCL library path for TensorRT distributed inference.

    This function:
    1. Detects if nvidia.nccl pip package is installed
    2. Creates libnccl.so symlink if needed
    3. Warns if LD_LIBRARY_PATH is not configured

    Call this before initializing NCCL communicators for TensorRT.

    For NGC containers (system NCCL), this is a no-op.
    For pip-installed PyTorch, this ensures TensorRT can find the correct NCCL.
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

    # Check LD_LIBRARY_PATH
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if nccl_lib_dir not in ld_library_path:
        logger.warning(
            f"\n"
            f"{'=' * 70}\n"
            f"NCCL LIBRARY PATH WARNING\n"
            f"{'=' * 70}\n"
            f"PyTorch's NCCL library directory is not in LD_LIBRARY_PATH.\n"
            f"TensorRT may load a different NCCL library than PyTorch,\n"
            f"causing distributed inference to fail.\n"
            f"\n"
            f"NCCL directory: {nccl_lib_dir}\n"
            f"\n"
            f"Please set before running:\n"
            f"  export LD_LIBRARY_PATH={nccl_lib_dir}:$LD_LIBRARY_PATH\n"
            f"{'=' * 70}\n"
        )
    else:
        logger.debug(f"LD_LIBRARY_PATH includes NCCL directory: {nccl_lib_dir}")

    if symlink_ok:
        logger.debug("NCCL library setup complete")
