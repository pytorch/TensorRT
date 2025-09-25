import ctypes
import getpass
import logging
import os
import platform
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

import torch
from torch.distributed._tensor.device_mesh import DeviceMesh, init_device_mesh
from torch_tensorrt._version import __tensorrt_llm_version__

_WHL_CPYTHON_VERSION = "cp310"

logger = logging.getLogger(__name__)


def check_tensor_parallel_device_number(world_size: int) -> None:
    if world_size % 2 != 0:
        raise ValueError(
            f"TP examples require even number of GPUs, but got {world_size} gpus"
        )


def get_tensor_parallel_device_mesh(
    rank: int = 0, world_size: int = 1
) -> tuple[DeviceMesh, int, int]:
    local_rank = int(
        os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", rank % torch.cuda.device_count())
    )
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", world_size))
    device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,))
    rank = device_mesh.get_rank()
    assert rank == local_rank
    device_id = (
        rank % torch.cuda.device_count()
    )  # Ensure each rank gets a unique device
    torch.cuda.set_device(device_id)

    return device_mesh, world_size, rank


def initialize_logger(rank: int, logger_file_name: str) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logger_file_name + f"_{rank}.log", mode="w")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    return logger


def is_platform_supported_for_trtllm() -> bool:
    """
    Checks if the current platform supports TensorRT-LLM plugins for the NCCL backend.

    Returns:
        bool: True if supported, False otherwise.

    Unsupported:
        - Windows platforms
        - Jetson/Orin/Xavier (aarch64 architecture + 'tegra' in platform release)
        - CUDA 13 not supported
    """
    system = platform.system().lower()
    machine = platform.machine().lower()
    release = platform.release().lower()

    if "windows" in system:
        logger.info(
            "TensorRT-LLM plugins for NCCL backend are not supported on Windows."
        )
        return False

    if machine == "aarch64" and "tegra" in release:
        logger.info(
            "TensorRT-LLM plugins for NCCL backend are not supported on Jetson/Orin/Xavier (Tegra) devices."
        )
        return False

    try:
        cuda_version = torch.version.cuda  # e.g., "12.4" or "13.0"
        if cuda_version is None:
            logger.warning("No CUDA runtime detected — TRT-LLM plugins unavailable.")
            return False

        major, minor = map(int, cuda_version.split("."))
        if major != 12:
            logger.warning("CUDA 13 is not supported for TRT-LLM plugins.")
            return False

        return True

    except Exception as e:
        logger.warning(f"Failed to detect CUDA version: {e}")
        return False

    return True


def _cache_root() -> Path:
    username = getpass.getuser()
    return Path(tempfile.gettempdir()) / f"torch_tensorrt_{username}"


def _extracted_dir_trtllm(platform_system: str, platform_machine: str) -> Path:
    return (
        _cache_root()
        / "trtllm"
        / f"{__tensorrt_llm_version__}_{platform_system}_{platform_machine}"
    )


def extract_wheel_file(wheel_path: Path, extract_dir: Path) -> None:
    from torch.distributed import barrier, get_rank, is_initialized

    if not is_initialized():
        # Single process case, just unzip
        is_master = True
    else:
        is_master = get_rank() == 0  # only rank 0 does the unzip

    if is_master:
        try:
            import zipfile
        except ImportError as e:
            raise ImportError(
                "zipfile module is required but not found. Please install zipfile"
            )
        try:
            with zipfile.ZipFile(wheel_path) as zip_ref:
                zip_ref.extractall(extract_dir)
                logger.debug(f"Extracted wheel to {extract_dir}")

        except FileNotFoundError as e:
            # This should capture the errors in the download failure above
            logger.error(f"Wheel file not found at {wheel_path}: {e}")
            raise RuntimeError(
                f"Failed to find downloaded wheel file at {wheel_path}"
            ) from e
        except zipfile.BadZipFile as e:
            logger.error(f"Invalid or corrupted wheel file: {e}")
            raise RuntimeError(
                "Downloaded wheel file is corrupted or not a valid zip archive"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error while extracting wheel: {e}")
            raise RuntimeError(
                "Unexpected error during extraction of TensorRT-LLM wheel"
            ) from e

    # Make sure others wait until unzip is done
    if is_initialized():
        barrier()


def download_and_get_plugin_lib_path() -> Optional[str]:
    """
    Returns the path to the TensorRT‑LLM shared library, downloading and extracting if necessary.

    Args:
        platform (str): Platform identifier (e.g., 'linux_x86_64')

    Returns:
        Optional[str]: Path to shared library or None if operation fails.
    """
    platform_system = platform.system().lower()
    platform_machine = platform.machine().lower()
    wheel_filename = (
        f"tensorrt_llm-{__tensorrt_llm_version__}-{_WHL_CPYTHON_VERSION}-"
        f"{_WHL_CPYTHON_VERSION}-{platform_system}_{platform_machine}.whl"
    )
    wheel_path = _cache_root() / wheel_filename
    extract_dir = _extracted_dir_trtllm(platform_system, platform_machine)
    # else will never be met though
    lib_filename = (
        "libnvinfer_plugin_tensorrt_llm.so"
        if "linux" in platform_system
        else "libnvinfer_plugin_tensorrt_llm.dll"
    )
    # eg: /tmp/torch_tensorrt_<username>/trtllm/0.17.0.post1_linux_x86_64/tensorrt_llm/libs/libnvinfer_plugin_tensorrt_llm.so
    plugin_lib_path = extract_dir / "tensorrt_llm" / "libs" / lib_filename

    if plugin_lib_path.exists():
        return str(plugin_lib_path)

    wheel_path.parent.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    if not wheel_path.exists():
        base_url = "https://pypi.nvidia.com/tensorrt-llm/"
        download_url = base_url + wheel_filename
        try:
            logger.debug(f"Downloading {download_url} ...")
            urllib.request.urlretrieve(download_url, wheel_path)
            logger.debug("Download succeeded and TRT-LLM wheel is now present")
        except urllib.error.HTTPError as e:
            logger.error(
                f"HTTP error {e.code} when trying to download {download_url}: {e.reason}"
            )
        except urllib.error.URLError as e:
            logger.error(
                f"URL error when trying to download {download_url}: {e.reason}"
            )
        except OSError as e:
            logger.error(f"Local file write error: {e}")

    extract_wheel_file(wheel_path, extract_dir)

    try:
        wheel_path.unlink(missing_ok=True)
        logger.debug(f"Deleted wheel file: {wheel_path}")
    except Exception as e:
        logger.warning(f"Could not delete wheel file {wheel_path}: {e}")
    if not plugin_lib_path.exists():
        logger.error(
            f"Plugin library not found at expected location: {plugin_lib_path}"
        )
        return None

    return str(plugin_lib_path)


def load_and_initialize_trtllm_plugin(plugin_lib_path: str) -> bool:
    """
    Loads and initializes the TensorRT-LLM plugin from the given shared library path.

    Args:
        plugin_lib_path (str): Path to the shared TensorRT-LLM plugin library.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        handle = ctypes.CDLL(plugin_lib_path)
        logger.info(f"Successfully loaded plugin library: {plugin_lib_path}")
    except OSError as e_os_error:
        if "libmpi" in str(e_os_error):
            logger.warning(
                f"Failed to load libnvinfer_plugin_tensorrt_llm.so from {plugin_lib_path}, got error {e_os_error} (hint: libmpi.so is a necessary dependency; ensure that OpenMPI or MPICH is installed on your system)",
                exc_info=e_os_error,
            )
        else:
            logger.warning(
                f"Failed to load libnvinfer_plugin_tensorrt_llm.so from {plugin_lib_path}. "
                f"Ensure the path is correct and the library is compatible.",
                exc_info=e_os_error,
            )
        return False

    try:
        handle.initTrtLlmPlugins.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        handle.initTrtLlmPlugins.restype = ctypes.c_bool
    except AttributeError as e_plugin_unavailable:
        logger.warning(
            "Unable to initialize the TensorRT-LLM plugin library",
            exc_info=e_plugin_unavailable,
        )
        return False

    try:
        if handle.initTrtLlmPlugins(None, b"tensorrt_llm"):
            logger.info("TensorRT-LLM plugin successfully initialized")
            return True
        else:
            logger.warning("TensorRT-LLM plugin library failed in initialization")
            return False
    except Exception as e_initialization_error:
        logger.warning(
            "Exception occurred during TensorRT-LLM plugin library initialization",
            exc_info=e_initialization_error,
        )
        return False
    return False


def load_tensorrt_llm_for_nccl() -> bool:
    """
    Attempts to load the TensorRT-LLM plugin and initialize it.
    Either the env variable TRTLLM_PLUGINS_PATH can specify the path
    Or the user can specify USE_TRTLLM_PLUGINS as either of (1, true, yes, on) to download the TRT-LLM distribution and load it

    Returns:
        bool: True if the plugin was successfully loaded and initialized, False otherwise.
    """
    if not is_platform_supported_for_trtllm():
        return False
    plugin_lib_path = os.environ.get("TRTLLM_PLUGINS_PATH")

    if plugin_lib_path:
        return load_and_initialize_trtllm_plugin(plugin_lib_path)
    else:
        # this option can be used by user if TRTLLM_PLUGINS_PATH is not set by user
        use_trtllm_plugin = os.environ.get("USE_TRTLLM_PLUGINS", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if not use_trtllm_plugin:
            logger.warning(
                "Neither TRTLLM_PLUGIN_PATH is set nor is it directed to download the shared library. Please set either of the two to use TRT-LLM libraries in torchTRT"
            )
            return False

        plugin_lib_path = download_and_get_plugin_lib_path()
        return load_and_initialize_trtllm_plugin(plugin_lib_path)  # type: ignore[arg-type]
    return False
