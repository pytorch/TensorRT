import logging
import os
import tempfile
from logging.config import dictConfig
from typing import Any, List, Optional

import torch
from torch_tensorrt.dynamo.lowering import (
    ATEN_POST_LOWERING_PASSES,
    ATEN_PRE_LOWERING_PASSES,
)

_LOGGER = logging.getLogger("torch_tensorrt [TensorRT Conversion Context]")
GRAPH_LEVEL = 5
logging.addLevelName(GRAPH_LEVEL, "GRAPHS")

# Debugger States
DEBUG_FILE_DIR = tempfile.TemporaryDirectory().name
SAVE_ENGINE_PROFILE = False


class Debugger:
    def __init__(
        self,
        level: str,
        capture_fx_graph_before: Optional[List[str]] = None,
        capture_fx_graph_after: Optional[List[str]] = None,
        save_engine_profile: bool = False,
        logging_dir: Optional[str] = None,
    ):

        if level != "graphs" and (capture_fx_graph_after or save_engine_profile):
            _LOGGER.warning(
                "Capture FX Graph or Draw Engine Graph is only supported when level is 'graphs'"
            )

        if level == "debug":
            self.level = logging.DEBUG
        elif level == "info":
            self.level = logging.INFO
        elif level == "warning":
            self.level = logging.WARNING
        elif level == "error":
            self.level = logging.ERROR
        elif level == "internal_errors":
            self.level = logging.CRITICAL
        elif level == "graphs":
            self.level = GRAPH_LEVEL

        else:
            raise ValueError(
                f"Invalid level: {level}, allowed levels are: debug, info, warning, error, internal_errors, graphs"
            )

        self.capture_fx_graph_before = capture_fx_graph_before
        self.capture_fx_graph_after = capture_fx_graph_after
        global SAVE_ENGINE_PROFILE
        SAVE_ENGINE_PROFILE = save_engine_profile

        if logging_dir is not None:
            global DEBUG_FILE_DIR
            DEBUG_FILE_DIR = logging_dir
        os.makedirs(DEBUG_FILE_DIR, exist_ok=True)

    def __enter__(self) -> None:
        self.original_lvl = _LOGGER.getEffectiveLevel()
        self.rt_level = torch.ops.tensorrt.get_logging_level()
        dictConfig(self.get_config())

        if self.level == GRAPH_LEVEL:
            self.old_pre_passes, self.old_post_passes = (
                ATEN_PRE_LOWERING_PASSES.passes,
                ATEN_POST_LOWERING_PASSES.passes,
            )
            pre_pass_names = [p.__name__ for p in self.old_pre_passes]
            post_pass_names = [p.__name__ for p in self.old_post_passes]
            path = os.path.join(DEBUG_FILE_DIR, "lowering_passes_visualization")
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

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:

        dictConfig(self.get_default_config())
        torch.ops.tensorrt.set_logging_level(self.rt_level)
        if self.level == GRAPH_LEVEL and self.capture_fx_graph_after:
            ATEN_PRE_LOWERING_PASSES.passes, ATEN_POST_LOWERING_PASSES.passes = (
                self.old_pre_passes,
                self.old_post_passes,
            )

    def get_config(self) -> dict[str, Any]:
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
                    "level": self.level,
                    "class": "logging.FileHandler",
                    "filename": f"{DEBUG_FILE_DIR}/torch_tensorrt_logging.log",
                    "formatter": "standard",
                },
                "console": {
                    "level": self.level,
                    "class": "logging.StreamHandler",
                    "formatter": "brief",
                },
            },
            "loggers": {
                "": {  # root logger
                    "handlers": ["file", "console"],
                    "level": self.level,
                    "propagate": True,
                },
            },
            "force": True,
        }
        return config

    def get_default_config(self) -> dict[str, Any]:
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
