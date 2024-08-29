import logging
from typing import Any

from torch_tensorrt.dynamo.runtime import PythonTorchTensorRTModule, TorchTensorRTModule

logger = logging.getLogger(__name__)


class _WeightStreamingContextManager(object):
    """
    Helper class used to setup weight streaming budget
    """

    def __init__(self, module: PythonTorchTensorRTModule | TorchTensorRTModule) -> None:
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
        total_device_budget = sum(self.streamable_budget)
        # device_budget is -1 if there is no trt module
        device_budget = -1 if total_device_budget == 0 else total_device_budget
        super().__setattr__("device_budget", device_budget)
        super().__setattr__("total_device_budget", total_device_budget)

    def __enter__(self) -> "_WeightStreamingContextManager":
        return self

    def __exit__(self, *args: Any) -> None:
        for i, (name, rt_mod) in enumerate(self.rt_mods):
            rt_mod.set_weight_streaming_budget(self.streamable_budget[i])
            logger.debug(
                f"Disable weight streaming by setting size {self.streamable_budget[i]} for {name}"
            )

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "device_budget":
            value = self.set_streamable_weight_bytes(value)
        super().__setattr__(name, value)

    def set_streamable_weight_bytes(self, budget_bytes: int) -> int:
        ws_budget_bytes = 0
        total_bytes = self.total_device_budget
        if total_bytes == 0:
            logger.error(
                "streamable bytes are zero. Was module complied with enable_weight_streaming=True option?"
            )
            return -1
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


def weight_streaming(
    module: PythonTorchTensorRTModule | TorchTensorRTModule,
) -> _WeightStreamingContextManager:
    return _WeightStreamingContextManager(module)
