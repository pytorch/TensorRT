import contextlib
import functools
import logging
import os
import tempfile
from logging.config import dictConfig
from typing import Any, List, Optional
from unittest import mock

import torch
from torch_tensorrt.dynamo.debug._DebuggerConfig import DebuggerConfig
from torch_tensorrt.dynamo.debug._supports_debugger import (
    _DEBUG_ENABLED_CLS,
    _DEBUG_ENABLED_FUNCS,
)
from torch_tensorrt.dynamo.lowering import (
    ATEN_POST_LOWERING_PASSES,
    ATEN_PRE_LOWERING_PASSES,
)

_LOGGER = logging.getLogger("torch_tensorrt [TensorRT Conversion Context]")
GRAPH_LEVEL = 5
logging.addLevelName(GRAPH_LEVEL, "GRAPHS")


class Debugger:
    def __init__(
        self,
        log_level: str = "debug",
        capture_fx_graph_before: Optional[List[str]] = None,
        capture_fx_graph_after: Optional[List[str]] = None,
        save_engine_profile: bool = False,
        engine_builder_monitor: bool = True,
        logging_dir: str = tempfile.gettempdir(),
    ):

        os.makedirs(logging_dir, exist_ok=True)
        self.cfg = DebuggerConfig(
            log_level=log_level,
            save_engine_profile=save_engine_profile,
            engine_builder_monitor=engine_builder_monitor,
            logging_dir=logging_dir,
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

    def __enter__(self) -> None:
        self.original_lvl = _LOGGER.getEffectiveLevel()
        self.rt_level = torch.ops.tensorrt.get_logging_level()
        dictConfig(self.get_customized_logging_config())

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
            f.__kwdefaults__["_debugger_settings"] = self.cfg

        [
            self._context_stack.enter_context(
                mock.patch.object(
                    c,
                    "__init__",
                    functools.partialmethod(c.__init__, _debugger_settings=self.cfg),
                )
            )
            for c in _DEBUG_ENABLED_CLS
        ]

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:

        dictConfig(self.get_default_logging_config())
        torch.ops.tensorrt.set_logging_level(self.rt_level)
        if self.capture_fx_graph_before or self.capture_fx_graph_after:
            ATEN_PRE_LOWERING_PASSES.passes, ATEN_POST_LOWERING_PASSES.passes = (
                self.old_pre_passes,
                self.old_post_passes,
            )
        self.debug_file_dir = tempfile.TemporaryDirectory().name

        for f in _DEBUG_ENABLED_FUNCS:
            f.__kwdefaults__["_debugger_settings"] = None

        self._context_stack.close()

    def get_customized_logging_config(self) -> dict[str, Any]:
        config = {
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
                "file": {
                    "level": self.log_level,
                    "class": "logging.FileHandler",
                    "filename": f"{self.cfg.logging_dir}/torch_tensorrt_logging.log",
                    "formatter": "standard",
                },
                "console": {
                    "level": self.log_level,
                    "class": "logging.StreamHandler",
                    "formatter": "brief",
                },
            },
            "loggers": {
                "": {  # root logger
                    "handlers": ["file", "console"],
                    "level": self.log_level,
                    "propagate": True,
                },
            },
            "force": True,
        }
        return config

    def get_default_logging_config(self) -> dict[str, Any]:
        config = {
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
                    "level": self.original_lvl,
                    "class": "logging.StreamHandler",
                    "formatter": "brief",
                },
            },
            "loggers": {
                "": {  # root logger
                    "handlers": ["console"],
                    "level": self.original_lvl,
                    "propagate": True,
                },
            },
            "force": True,
        }
        return config
