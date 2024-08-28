import logging
from functools import wraps
from typing import Any

from torch_tensorrt.dynamo.runtime import PythonTorchTensorRTModule, TorchTensorRTModule

logger = logging.getLogger(__name__)


def recreate_context_decorator(method):
    """
    A decorator that destroys a context before a method execution and
    creates it after the method execution within the same class instance.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        # Destroy the context before the method execution
        self._reset_context()
        # Execute the method
        result = method(self, *args, **kwargs)
        # Re-create the context after the method execution
        self._init_context()
        return result

    return wrapper


class _WeightStreamingContextManager(object):
    """
    Helper class used to setup weight streaming budget
    """

    def __init__(self, module) -> None:
        rt_mods = []
        for name, rt_mod in module.named_children():
            if "_run_on_acc" in name and isinstance(
                rt_mod, (PythonTorchTensorRTModule, TorchTensorRTModule)
            ):
                rt_mods.append((name, rt_mod))
        self.streamable_budget = [
            mod.get_streamable_weights_size() for _, mod in rt_mods
        ]
        self.rt_mods = rt_mods

    def _reset_context(self):
        for _, rt_mod in self.rt_mods:
            rt_mod.reset_context()

    def _init_context(self):
        for _, rt_mod in self.rt_mods:
            rt_mod.init_context()

    def __enter__(self) -> "_WeightStreamingContextManager":
        return self

    @recreate_context_decorator
    def __exit__(self, *args: Any) -> None:
        for i, (name, rt_mod) in enumerate(self.rt_mods):
            rt_mod.set_weight_streaming_budget(self.streamable_budget[i])
            logger.debug(
                f"Disable weight streaming by setting size {self.streamable_budget[i]} for {name}"
            )

    def get_streamable_weight_bytes(self):
        return sum(self.streamable_budget)

    @recreate_context_decorator
    def set_streamable_weight_bytes(self, budget_bytes):
        ws_budget_bytes = 0
        total_bytes = self.get_streamable_weight_bytes()
        if total_bytes == 0:
            logger.error(
                "streamable bytes are zero. Was module complied with enable_weight_streaming=True option?"
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

    @recreate_context_decorator
    def set_automatic_streaming_budget(self):
        total_bytes = 0
        for _, rt_mod in self.rt_mods:
            total_bytes += rt_mod.set_automatic_streaming_budget()

        return total_bytes


def enable_weight_streaming(module) -> _WeightStreamingContextManager:
    return _WeightStreamingContextManager(module)
