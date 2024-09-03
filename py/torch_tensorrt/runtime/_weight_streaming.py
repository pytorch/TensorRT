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
        trt_min_budget = 0
        trt_max_budget = 0
        current_trt_budget = 0
        for name, rt_mod in module.named_children():
            if "_run_on_acc" in name and isinstance(
                rt_mod, (PythonTorchTensorRTModule, TorchTensorRTModule)
            ):
                trt_min_budget += rt_mod.get_min_required_device_budget()
                trt_max_budget += rt_mod.get_streamable_weights_size()
                current_trt_budget += rt_mod.get_weight_streaming_budget()
                rt_mods.append((name, rt_mod))
            else:
                torch_budget += sum(
                    [p.numel() * p.element_size() for p in rt_mod.parameters()]
                )

        trt_max_budget += trt_min_budget
        self.torch_budget = torch_budget
        self.rt_mods = rt_mods
        device_budget = torch_budget + trt_min_budget + current_trt_budget
        super().__setattr__("device_budget", device_budget)
        super().__setattr__("trt_max_budget", trt_max_budget)

    def get_automatic_weight_streaming_budget(self) -> int:
        ws_budget_bytes = self.torch_budget
        for _, rt_mod in self.rt_mods:
            ws_budget_bytes += rt_mod.get_automatic_weight_streaming_budget()
            ws_budget_bytes += rt_mod.get_min_required_device_budget()
        return ws_budget_bytes

    def get_required_device_budgets(self) -> tuple[int, int]:
        min_budget = self.torch_budget
        max_budget = self.torch_budget + self.trt_max_budget
        for _, rt_mod in self.rt_mods:
            min_budget += rt_mod.get_min_required_device_budget()
        return min_budget, max_budget

    def __enter__(self) -> "_WeightStreamingContextManager":
        return self

    def __exit__(self, *args: Any) -> None:
        max_budget = self.torch_budget + self.trt_max_budget
        if self.trt_max_budget > 0:
            logger.debug(
                f"Disable weight streaming by applying max budget size {max_budget}"
            )
            self.device_budget = max_budget

    def _set_streamable_weight_bytes(self, requested_budget: int) -> int:
        ws_budget_bytes = self.torch_budget
        trt_engine_budget = requested_budget - self.torch_budget
        if self.trt_max_budget == 0:
            raise RuntimeError(
                "Streamable bytes are zero. Was module complied with enable_weight_streaming=True option?"
            )
        elif trt_engine_budget <= 0:
            raise RuntimeError(
                f"Requested budget {requested_budget} is less than mininum torch budget: {self.torch_budget}"
            )
        else:
            # Normalized size is applied for multiple trt runtime module.
            # e.g. 100B budget is applied to two modules and they have 1000B and 3000B max streamable size respectively.
            # Then 25B and 75B are applied for each module.
            for mod_name, rt_mod in self.rt_mods:
                max_budget = (
                    rt_mod.get_min_required_device_budget()
                    + rt_mod.get_streamable_weights_size()
                )
                normalized_size = (
                    int(max_budget / self.trt_max_budget * trt_engine_budget)
                    - rt_mod.get_min_required_device_budget()
                )

                if normalized_size < 0:
                    raise RuntimeError(
                        f"Requested trt budget {trt_engine_budget} is less than mininum trt budget of submodule {mod_name} size={rt_mod.get_min_required_device_budget()}"
                    )

                ws_budget_bytes += rt_mod.set_device_memory_budget(normalized_size)
                ws_budget_bytes += rt_mod.get_min_required_device_budget()
                logger.debug(
                    f"Set weight streaming size {normalized_size} for {mod_name}"
                )
        return ws_budget_bytes

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "device_budget":
            value = self._set_streamable_weight_bytes(value)
        super().__setattr__(name, value)


def weight_streaming(
    module: torch.fx.GraphModule,
) -> _WeightStreamingContextManager:
    return _WeightStreamingContextManager(module)
