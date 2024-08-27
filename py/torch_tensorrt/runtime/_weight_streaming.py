import logging
from typing import Any

import torch
import tensorrt
import torch_tensorrt

from torch_tensorrt.dynamo.runtime import PythonTorchTensorRTModule, TorchTensorRTModule

logger = logging.getLogger(__name__)


class _WeightStreamingContextManager(object):
    """
    Helper class used to setup weight streaming budget
    """

    def __init__(self, module) -> None:
        rt_mods = []
        for name, rt_mod in module.named_children():
            if "_run_on_acc" in name and (
                isinstance(rt_mod, PythonTorchTensorRTModule)
                or isinstance(rt_mod, TorchTensorRTModule)
            ):
                rt_mods.append((name, rt_mod))
        self.streamable_budget = [
            mod.get_weight_streaming_budget() for _, mod in rt_mods
        ]
        self.rt_mods = rt_mods

    def __enter__(self) -> "_WeightStreamingContextManager":
        return self

    def __exit__(self, *args: Any) -> None:
        for _, rt_mod in self.rt_mods:
            rt_mod.reset_context()

    def get_streamable_weight_bytes(self):
        return sum(self.streamable_budget)

    def set_streamable_weight_bytes(self, budget_bytes):
        ws_budget_bytes = 0
        total_bytes = self.get_streamable_weight_bytes()
        if total_bytes == 0:
            logger.error(
                f"streamable bytes are zero. Was module complied with enable_weight_streaming=True option?"
            )
            return 0
        elif total_bytes <= budget_bytes:
            logger.error(
                f"Requested budget is equal or greater than streamable bytes: {total_bytes}"
            )
        # Normalized size is applied for multiple runtime module.
        # e.g. 100B budget is applied to two modules and they have 1000B and 3000B streamable size respectively.
        # Then 25B and 75B are applied for each module.
        normalized_size = [
            int(streamable_bytes / total_bytes * budget_bytes)
            for streamable_bytes in self.streamable_budget
        ]
        for i, (name, rt_mod) in enumerate(self.rt_mods):
            ws_budget_bytes += rt_mod.set_weight_streaming_budget(normalized_size[i])
            logger.debug(f"Set weight streaming size {normalized_size[i]} for {name}")
        return ws_budget_bytes

    def set_automatic_streaming_budget(self):
        total_bytes = 0
        for _, rt_mod in self.rt_mods:
            total_bytes += rt_mod.set_automatic_streaming_budget()

        return total_bytes


def enable_weight_streaming(module) -> _WeightStreamingContextManager:
    return _WeightStreamingContextManager(module)
