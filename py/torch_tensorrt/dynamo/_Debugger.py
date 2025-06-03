from typing import Any, List, Optional, Dict
import copy
import functools
import contextlib

from unittest import mock
from dataclasses import dataclass
import logging
import os
import tempfile
from logging.config import dictConfig

import torch
import torch_tensorrt
from torch_tensorrt.dynamo._DebuggerConfig import DebuggerConfig
from torch_tensorrt.dynamo._supports_debugger import _DEBUG_ENABLED_CLS, _DEBUG_ENABLED_FUNCS
from torch_tensorrt.dynamo.conversion._TRTInterpreter import TRTInterpreter
from torch_tensorrt.dynamo.lowering.passes.constant_folding import constant_fold
from torch_tensorrt.dynamo.lowering import (
    ATEN_POST_LOWERING_PASSES,
    ATEN_PRE_LOWERING_PASSES,
)

_LOGGER = logging.getLogger("torch_tensorrt [TensorRT Conversion Context]")
GRAPH_LEVEL = 5
logging.addLevelName(GRAPH_LEVEL, "GRAPH")

# Debugger States


class Debugger:
    def __init__(self, **kwargs: Dict[str, Any]):
        self.cfg = DebuggerConfig(**kwargs)

    def __enter__(self) -> None:
        self.original_lvl = _LOGGER.getEffectiveLevel()
        self.rt_level = torch.ops.tensorrt.get_logging_level()
        #dictConfig(self.get_config())

        self.old_pre_passes, self.old_post_passes = (
            ATEN_PRE_LOWERING_PASSES.passes,
            ATEN_POST_LOWERING_PASSES.passes,
        )
        pre_pass_names = [p.__name__ for p in self.old_pre_passes]
        post_pass_names = [p.__name__ for p in self.old_post_passes]
        #path = os.path.join(DEBUG_FILE_DIR, "lowering_passes_visualization")

        ATEN_PRE_LOWERING_PASSES.insert_debug_pass_before(self.cfg.capture_fx_graph_before)
        ATEN_POST_LOWERING_PASSES.insert_debug_pass_before(self.cfg.capture_fx_graph_before)

        ATEN_PRE_LOWERING_PASSES.insert_debug_pass_after(self.cfg.capture_fx_graph_after)
        ATEN_POST_LOWERING_PASSES.insert_debug_pass_after(self.cfg.capture_fx_graph_after)

        self._context_stack = contextlib.ExitStack()

        for f in _DEBUG_ENABLED_FUNCS:
            f.__kwdefaults__["_debugger_settings"] = self.cfg

        [
            self._context_stack.enter_context(
                mock.patch.object(
                    c,
                    '__init__',
                    functools.partialmethod(
                        c.__init__,
                        _debugger_settings=self.cfg
                    )
                )
            ) for c in _DEBUG_ENABLED_CLS
        ]

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        #dictConfig(self.get_default_config())
        torch.ops.tensorrt.set_logging_level(self.rt_level)

        ATEN_PRE_LOWERING_PASSES.passes, ATEN_POST_LOWERING_PASSES.passes = (
            self.old_pre_passes,
            self.old_post_passes,
        )

        for f in _DEBUG_ENABLED_FUNCS:
            f.__kwdefaults__["_debugger_settings"] = None

        self._context_stack.close()



    # def get_config(self) -> dict[str, Any]:
    #     config = {
    #         "version": 1,
    #         "disable_existing_loggers": False,
    #         "formatters": {
    #             "brief": {
    #                 "format": "%(asctime)s - %(levelname)s - %(message)s",
    #                 "datefmt": "%H:%M:%S",
    #             },
    #             "standard": {
    #                 "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    #                 "datefmt": "%Y-%m-%d %H:%M:%S",
    #             },
    #         },
    #         "handlers": {
    #             "file": {
    #                 "level": self.level,
    #                 "class": "logging.FileHandler",
    #                 "filename": f"{DEBUG_FILE_DIR}/torch_tensorrt_logging.log",
    #                 "formatter": "standard",
    #             },
    #             "console": {
    #                 "level": self.level,
    #                 "class": "logging.StreamHandler",
    #                 "formatter": "brief",
    #             },
    #         },
    #         "loggers": {
    #             "": {  # root logger
    #                 "handlers": ["file", "console"],
    #                 "level": self.level,
    #                 "propagate": True,
    #             },
    #         },
    #         "force": True,
    #     }
    #     return config

    # def get_default_config(self) -> dict[str, Any]:
    #     config = {
    #         "version": 1,
    #         "disable_existing_loggers": False,
    #         "formatters": {
    #             "brief": {
    #                 "format": "%(asctime)s - %(levelname)s - %(message)s",
    #                 "datefmt": "%H:%M:%S",
    #             },
    #             "standard": {
    #                 "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    #                 "datefmt": "%Y-%m-%d %H:%M:%S",
    #             },
    #         },
    #         "handlers": {
    #             "console": {
    #                 "level": self.original_lvl,
    #                 "class": "logging.StreamHandler",
    #                 "formatter": "brief",
    #             },
    #         },
    #         "loggers": {
    #             "": {  # root logger
    #                 "handlers": ["console"],
    #                 "level": self.original_lvl,
    #                 "propagate": True,
    #             },
    #         },
    #         "force": True,
    #     }
    #     return config
