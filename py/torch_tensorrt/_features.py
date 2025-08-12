import importlib
import os
import sys
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from torch_tensorrt._utils import (
    check_cross_compile_trt_win_lib,
    is_tensorrt_rtx,
    sanitized_torch_version,
)

from packaging import version

FeatureSet = namedtuple(
    "FeatureSet",
    [
        "torchscript_frontend",
        "torch_tensorrt_runtime",
        "dynamo_frontend",
        "fx_frontend",
        "refit",
        "qdp_plugin",
        "windows_cross_compile",
    ],
)

trtorch_dir = os.path.dirname(__file__)
linked_file = os.path.join(
    "lib", "torchtrt.dll" if sys.platform.startswith("win") else "libtorchtrt.so"
)
linked_file_runtime = os.path.join(
    "lib",
    (
        "torchtrt_runtime.dll"
        if sys.platform.startswith("win")
        else "libtorchtrt_runtime.so"
    ),
)
linked_file_full_path = os.path.join(trtorch_dir, linked_file)
linked_file_runtime_full_path = os.path.join(trtorch_dir, linked_file_runtime)

_TS_FE_AVAIL = os.path.isfile(linked_file_full_path)
_TORCHTRT_RT_AVAIL = _TS_FE_AVAIL or os.path.isfile(linked_file_runtime_full_path)
_DYNAMO_FE_AVAIL = version.parse(sanitized_torch_version()) >= version.parse("2.1.dev")
_FX_FE_AVAIL = False if is_tensorrt_rtx() else True
_REFIT_AVAIL = True
_WINDOWS_CROSS_COMPILE = check_cross_compile_trt_win_lib()

if importlib.util.find_spec("tensorrt.plugin"):
    _QDP_PLUGIN_AVAIL = True
else:
    _QDP_PLUGIN_AVAIL = False

ENABLED_FEATURES = FeatureSet(
    _TS_FE_AVAIL,
    _TORCHTRT_RT_AVAIL,
    _DYNAMO_FE_AVAIL,
    _FX_FE_AVAIL,
    _REFIT_AVAIL,
    _QDP_PLUGIN_AVAIL,
    _WINDOWS_CROSS_COMPILE,
)

T = TypeVar("T")


def _enabled_features_str() -> str:
    enabled = lambda x: "ENABLED" if x else "DISABLED"
    out_str: str = f"Enabled Features:\n - Dynamo Frontend: {enabled(_DYNAMO_FE_AVAIL)}\n - Torch-TensorRT Runtime: {enabled(_TORCHTRT_RT_AVAIL)}\n - FX Frontend: {enabled(_FX_FE_AVAIL)}\n - TorchScript Frontend: {enabled(_TS_FE_AVAIL)}\n - Refit: {enabled(_REFIT_AVAIL)}\n - QDP Plugin: {enabled(_QDP_PLUGIN_AVAIL)}\n"  # type: ignore[no-untyped-call]
    return out_str


def needs_torch_tensorrt_runtime(f: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: List[Any], **kwargs: Dict[str, Any]) -> Any:
        if ENABLED_FEATURES.torch_tensorrt_runtime:
            return f(*args, **kwargs)
        else:

            def not_implemented(*args: List[Any], **kwargs: Dict[str, Any]) -> Any:
                raise NotImplementedError("Torch-TensorRT Runtime is not available")

            return not_implemented(*args, **kwargs)

    return wrapper


def needs_qdp_plugin(f: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: List[Any], **kwargs: Dict[str, Any]) -> Any:
        if ENABLED_FEATURES.qdp_plugin:
            return f(*args, **kwargs)
        else:

            def not_implemented(*args: List[Any], **kwargs: Dict[str, Any]) -> Any:
                raise NotImplementedError(
                    "TensorRT QDP(Quick Deploy Plugins) not available, requires TensorRT 10.7.0 or higher"
                )

            return not_implemented(*args, **kwargs)

    return wrapper


def needs_refit(f: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: List[Any], **kwargs: Dict[str, Any]) -> Any:
        if ENABLED_FEATURES.refit:
            return f(*args, **kwargs)
        else:

            def not_implemented(*args: List[Any], **kwargs: Dict[str, Any]) -> Any:
                raise NotImplementedError(
                    "Refit feature is currently not available in Python 3.13 or higher"
                )

            return not_implemented(*args, **kwargs)

    return wrapper


def needs_cross_compile(f: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: List[Any], **kwargs: Dict[str, Any]) -> Any:
        if ENABLED_FEATURES.windows_cross_compile:
            return f(*args, **kwargs)
        else:

            def not_implemented(*args: List[Any], **kwargs: Dict[str, Any]) -> Any:
                raise NotImplementedError(
                    "Windows cross compilation feature is not available"
                )

            return not_implemented(*args, **kwargs)

    return wrapper


def for_all_methods(
    decorator: Callable[..., Any], exclude: Optional[List[str]] = None
) -> Callable[..., Any]:
    exclude_list: List[str] = []
    if exclude:
        exclude_list = exclude

    def decorate(cls: Type[T]) -> Type[T]:
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)) and attr not in exclude_list:
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls

    return decorate
