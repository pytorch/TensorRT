import contextlib
import functools
import logging
import os
import sys
import tempfile
from logging.config import dictConfig
from typing import Any, List, Optional
from unittest import mock

import torch
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt._utils import is_tensorrt_version_supported
from torch_tensorrt.dynamo._defaults import DEBUG_LOGGING_DIR
from torch_tensorrt.dynamo.debug._DebuggerConfig import DebuggerConfig
from torch_tensorrt.dynamo.debug._supports_debugger import (
    _DEBUG_ENABLED_CLS,
    _DEBUG_ENABLED_FUNCS,
)
from torch_tensorrt.dynamo.lowering import (
    ATEN_POST_LOWERING_PASSES,
    ATEN_PRE_LOWERING_PASSES,
)

_LOGGER = logging.getLogger(__name__)
GRAPH_LEVEL = 5
logging.addLevelName(GRAPH_LEVEL, "GRAPHS")


class Debugger:
    def __init__(
        self,
        log_level: str = "debug",
        capture_fx_graph_before: Optional[List[str]] = None,
        capture_fx_graph_after: Optional[List[str]] = None,
        save_engine_profile: bool = False,
        capture_shim: bool = False,
        profile_format: str = "perfetto",
        engine_builder_monitor: bool = True,
        logging_dir: str = DEBUG_LOGGING_DIR,
        save_layer_info: bool = False,
    ):
        """Initialize a debugger for TensorRT conversion.

        Args:
            log_level (str): Logging level to use. Valid options are:
                'debug', 'info', 'warning', 'error', 'internal_errors', 'graphs'.
                Defaults to 'debug'.
            capture_fx_graph_before (List[str], optional): List of pass names to visualize FX graph
                before execution of a lowering pass. Defaults to None.
            capture_fx_graph_after (List[str], optional): List of pass names to visualize FX graph
                after execution of a lowering pass. Defaults to None.
            save_engine_profile (bool): Whether to save TensorRT engine profiling information.
                Defaults to False.
            capture_shim (bool): Whether to enable the capture shim feature. It is part of the TensorRT capture and replay feature, the captured output will be able to replay for debug purpose.
                Defaults to False.
            profile_format (str): Format for profiling data. Choose from 'perfetto', 'trex', 'cudagraph'.
                If you need to generate engine graph using the profiling files, set it to 'trex' and use the C++ runtime.
                If you need to generate cudagraph visualization, set it to 'cudagraph'.
                Defaults to 'perfetto'.
            engine_builder_monitor (bool): Whether to monitor TensorRT engine building process.
                Defaults to True.
            logging_dir (str): Directory to save debug logs and profiles.
                Defaults to system temp directory.
            save_layer_info (bool): Whether to save layer info.
                Defaults to False.
        """

        os.makedirs(logging_dir, exist_ok=True)
        self.cfg = DebuggerConfig(
            log_level=log_level,
            save_engine_profile=save_engine_profile,
            capture_shim=capture_shim,
            engine_builder_monitor=engine_builder_monitor,
            logging_dir=logging_dir,
            profile_format=profile_format,
            save_layer_info=save_layer_info,
        )

        if log_level == "debug":
            self.log_level = logging.DEBUG
        elif log_level == "info":
            self.log_level = logging.INFO
        elif log_level == "warning":
            self.log_level = logging.WARNING
        elif log_level == "error":
            self.log_level = logging.ERROR
        elif log_level == "internal_errors":
            self.log_level = logging.CRITICAL
        elif log_level == "graphs":
            self.log_level = GRAPH_LEVEL

        else:
            raise ValueError(
                f"Invalid level: {log_level}, allowed levels are: debug, info, warning, error, internal_errors, graphs"
            )

        self.capture_fx_graph_before = capture_fx_graph_before
        self.capture_fx_graph_after = capture_fx_graph_after

        if self.cfg.capture_shim:
            if not sys.platform.startswith("linux"):
                _LOGGER.warning(
                    "capture_shim featureis only supported on linux, will disable it"
                )
                self.cfg.capture_shim = False
                return
            if ENABLED_FEATURES.tensorrt_rtx:
                _LOGGER.warning(
                    "capture_shim feature is not supported on TensorRT-RTX, will disable it"
                )
                self.cfg.capture_shim = False
                return
            if not is_tensorrt_version_supported("10.13.0"):
                _LOGGER.warning(
                    "capture_shim feature is only supported on TensorRT 10.13 and above, will disable it"
                )
                self.cfg.capture_shim = False
                return

    def __enter__(self) -> None:
        self.original_lvl = _LOGGER.getEffectiveLevel()
        if ENABLED_FEATURES.torch_tensorrt_runtime:
            self.rt_level = torch.ops.tensorrt.get_logging_level()
        dictConfig(self.get_logging_config(self.log_level))

        if self.capture_fx_graph_before or self.capture_fx_graph_after:
            self.old_pre_passes, self.old_post_passes = (
                ATEN_PRE_LOWERING_PASSES.passes,
                ATEN_POST_LOWERING_PASSES.passes,
            )
            pre_pass_names = [p.__name__ for p in self.old_pre_passes]
            post_pass_names = [p.__name__ for p in self.old_post_passes]
            path = os.path.join(self.cfg.logging_dir, "lowering_passes_visualization")
            if self.capture_fx_graph_before is not None:
                pre_vis_passes = [
                    p for p in self.capture_fx_graph_before if p in pre_pass_names
                ]
                post_vis_passes = [
                    p for p in self.capture_fx_graph_before if p in post_pass_names
                ]
                ATEN_PRE_LOWERING_PASSES.insert_debug_pass_before(pre_vis_passes, path)
                ATEN_POST_LOWERING_PASSES.insert_debug_pass_before(
                    post_vis_passes, path
                )
            if self.capture_fx_graph_after is not None:
                pre_vis_passes = [
                    p for p in self.capture_fx_graph_after if p in pre_pass_names
                ]
                post_vis_passes = [
                    p for p in self.capture_fx_graph_after if p in post_pass_names
                ]
                ATEN_PRE_LOWERING_PASSES.insert_debug_pass_after(pre_vis_passes, path)
                ATEN_POST_LOWERING_PASSES.insert_debug_pass_after(post_vis_passes, path)

        self._context_stack = contextlib.ExitStack()

        for f in _DEBUG_ENABLED_FUNCS:
            f.__kwdefaults__["_debugger_config"] = self.cfg

        [
            self._context_stack.enter_context(
                mock.patch.object(
                    c,
                    "__init__",
                    functools.partialmethod(c.__init__, _debugger_config=self.cfg),
                )
            )
            for c in _DEBUG_ENABLED_CLS
        ]

        if self.cfg.capture_shim:
            shim_lib_name = "libtensorrt_shim.so"
            nvinfer_lib_name = "libnvinfer.so"

            def validate_setting() -> bool:
                is_valid = True
                # LD_PRELOAD and TRT_SHIM_NVINFER_LIB_NAME only read at exec-time; setting it during a running process won’t interpose already-loaded libs.
                # so, must set them before the tensorrt is loaded, cannot set during the Debugger.__enter__
                if os.environ.get("LD_PRELOAD") is None:
                    _LOGGER.error(
                        f"LD_PRELOAD is not set, please add the {shim_lib_name} with full path to the LD_PRELOAD environment variable"
                    )
                    is_valid = False
                if os.environ.get("TRT_SHIM_NVINFER_LIB_NAME") is None:
                    _LOGGER.error(
                        f"TRT_SHIM_NVINFER_LIB_NAME is not set, please add the {nvinfer_lib_name} with full path to the TRT_SHIM_NVINFER_LIB_NAME environment variable"
                    )
                    is_valid = False
                if os.environ.get("TRT_SHIM_OUTPUT_JSON_FILE") is None:
                    _LOGGER.error(
                        "TRT_SHIM_OUTPUT_JSON_FILE is not set, please add the shim output json file name with full path to the TRT_SHIM_OUTPUT_JSON_FILE environment variable"
                    )
                    is_valid = False
                else:
                    shim_output_json_file = os.environ["TRT_SHIM_OUTPUT_JSON_FILE"]
                    shim_output_dir = os.path.dirname(shim_output_json_file)
                    if len(shim_output_dir) > 0 and not os.path.exists(shim_output_dir):
                        _LOGGER.debug(
                            f"shim output directory {shim_output_dir} does not exist, creating it now"
                        )
                        os.makedirs(shim_output_dir)
                return is_valid

            if not validate_setting():
                return
            json_file_name = os.environ["TRT_SHIM_OUTPUT_JSON_FILE"]
            _LOGGER.info(
                f"capture_shim feature is enabled, shim output file is set to {json_file_name}"
            )

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:

        dictConfig(self.get_logging_config(None))
        if ENABLED_FEATURES.torch_tensorrt_runtime:
            torch.ops.tensorrt.set_logging_level(self.rt_level)
        if self.capture_fx_graph_before or self.capture_fx_graph_after:
            ATEN_PRE_LOWERING_PASSES.passes, ATEN_POST_LOWERING_PASSES.passes = (
                self.old_pre_passes,
                self.old_post_passes,
            )
        self.debug_file_dir = tempfile.TemporaryDirectory().name

        for f in _DEBUG_ENABLED_FUNCS:
            f.__kwdefaults__["_debugger_config"] = None

        self._context_stack.close()

    def get_logging_config(self, log_level: Optional[int] = None) -> dict[str, Any]:
        level = log_level if log_level is not None else self.original_lvl
        config: dict[str, Any] = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "brief": {
                    "format": "%(asctime)s - %(levelname)s - %(message)s",
                    "datefmt": "%H:%M:%S",
                },
                "standard": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "level": level,
                    "class": "logging.StreamHandler",
                    "formatter": "brief",
                },
            },
            "loggers": {
                "": {  # root logger
                    "handlers": ["console"],
                    "level": level,
                    "propagate": True,
                },
            },
            "force": True,
        }
        if log_level is not None:
            config["handlers"]["file"] = {
                "level": level,
                "class": "logging.FileHandler",
                "filename": f"{self.cfg.logging_dir}/torch_tensorrt_logging.log",
                "formatter": "standard",
            }
            config["loggers"][""]["handlers"].append("file")
        return config
