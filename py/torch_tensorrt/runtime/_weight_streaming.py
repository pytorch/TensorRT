import logging
from typing import Any

import torch
from torch_tensorrt.dynamo.runtime import PythonTorchTensorRTModule, TorchTensorRTModule

logger = logging.getLogger(__name__)


class _WeightStreamingContextManager(object):
    """
    Helper class used to setup weight streaming budget
    """

    def __init__(self, module: torch.fx.GraphModule) -> None:
        rt_mods = []
        torch_budget = 0
        trt_budget = 0
        for name, rt_mod in module.named_children():
            if "_run_on_acc" in name and isinstance(
                rt_mod, (PythonTorchTensorRTModule, TorchTensorRTModule)
            ):
                trt_budget += rt_mod.min_required_device_budget
                trt_budget += rt_mod.get_streamable_weights_size()
                rt_mods.append((name, rt_mod))
            else:
                torch_budget += sum(
                    [p.numel() * p.element_size() for p in rt_mod.parameters()]
                )
        self.torch_budget = torch_budget
        self.rt_mods = rt_mods
        total_device_budget = torch_budget + trt_budget
        # device_budget is -1 if there is no trt module
        device_budget = -1 if trt_budget == 0 else total_device_budget
        super().__setattr__("device_budget", device_budget)
        super().__setattr__("total_trt_budget", trt_budget)

    def get_min_required_device_budget(self) -> int:
        min_budget = self.torch_budget
        for _, rt_mod in self.rt_mods:
            min_budget += rt_mod.min_required_device_budget
        return min_budget

    def __enter__(self) -> "_WeightStreamingContextManager":
        return self

    def __exit__(self, *args: Any) -> None:
        for name, rt_mod in self.rt_mods:
            streamable_budget = rt_mod.get_streamable_weights_size()
            rt_mod.set_device_memory_budget(streamable_budget)
            logger.debug(
                f"Disable weight streaming by setting size {streamable_budget} for {name}"
            )

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "device_budget":
            requested_budget = value
            trt_engine_budget = requested_budget - self.torch_budget
            value = 0
            if self.total_trt_budget == 0:
                logger.error(
                    "Streamable bytes are zero. Was module complied with enable_weight_streaming=True option?"
                )
                value = -1
            elif trt_engine_budget <= 0:
                logger.error(
                    f"Requested budget {requested_budget} is less than mininum torch budget: {self.torch_budget}"
                )
                value = -1
            else:
                # Normalized size is applied for multiple trt runtime module.
                # e.g. 100B budget is applied to two modules and they have 1000B and 3000B max streamable size respectively.
                # Then 25B and 75B are applied for each module.
                for mod_name, rt_mod in self.rt_mods:
                    max_budget = (
                        rt_mod.min_required_device_budget
                        + rt_mod.get_streamable_weights_size()
                    )
                    normalized_size = (
                        int(max_budget / self.total_trt_budget * trt_engine_budget)
                        - rt_mod.min_required_device_budget
                    )
                    if normalized_size < 0:
                        logger.error(
                            f"Requested trt budget {trt_engine_budget} is less than mininum trt budget: {rt_mod.min_required_device_budget}"
                        )
                        value = -1
                        break
                    value += rt_mod.set_device_memory_budget(normalized_size)
                    value += rt_mod.min_required_device_budget
                    logger.debug(
                        f"Set weight streaming size {normalized_size} for {mod_name}"
                    )

        super().__setattr__(name, value)


def weight_streaming(
    module: torch.fx.GraphModule,
) -> _WeightStreamingContextManager:
    return _WeightStreamingContextManager(module)
