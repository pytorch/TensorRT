import logging
from typing import Any

import torch
import tensorrt
import torch_tensorrt

from torch_tensorrt.dynamo.runtime import PythonTorchTensorRTModule, TorchTensorRTModule

logger = logging.getLogger(__name__)


class WeightStreamingBudget(object):
    def __init__(self, name, rt_mod) -> None:
        self.name = name
        self.rt_mod = rt_mod
        self.engine = rt_mod.engine

    def __new__(cls, *args):
        if tensorrt.__version__ < "10.1":
            return super().__new__(WeightStreamingBudgetV1)
        else:
            return super().__new__(WeightStreamingBudgetV2)

    def get_weight_streaming_budget(self):
        pass

    def set_weight_streaming_budget(self, budget_bytes):
        pass

    def set_automatic_streaming_budget(self):
        pass

    def reset_context(self):
        self.rt_mod.reset_context()


class WeightStreamingBudgetV1(WeightStreamingBudget):
    def __init__(self, name, rt_mod) -> None:
        super().__init__(name, rt_mod)

    def get_weight_streaming_budget(self):
        print("todo")

    def set_weight_streaming_budget(self, budget_bytes):
        print("todo")


class WeightStreamingBudgetV2(WeightStreamingBudget):
    def __init__(self, name, rt_mod) -> None:
        super().__init__(name, rt_mod)
        self.streamable_weights_size = self.engine.streamable_weights_size

    def get_weight_streaming_budget(self):
        return self.streamable_weights_size

    def set_weight_streaming_budget(self, budget_bytes):
        self.engine.weight_streaming_budget_v2 = budget_bytes
        if self.engine.weight_streaming_budget_v2 != budget_bytes:
            RuntimeError(f"Failed to set weight streaming budget to {budget_bytes}!")

    def set_automatic_streaming_budget(self):
        budget_bytes = self.engine.get_weight_streaming_automatic_budget()
        self.engine.weight_streaming_budget_v2 = budget_bytes
        if self.engine.weight_streaming_budget_v2 != budget_bytes:
            RuntimeError(f"Failed to set weight streaming budget to {budget_bytes}!")


class _WeightStreamingContextManager(object):
    """
    Helper class used to setup weight streaming budget
    """

    def __init__(self, module) -> None:
        ws_budget = []
        for name, rt_mod in module.named_children():
            if "_run_on_acc" in name and (
                isinstance(rt_mod, PythonTorchTensorRTModule)
                or isinstance(rt_mod, TorchTensorRTModule)
            ):
                ws_budget.append(WeightStreamingBudget(name, rt_mod))
        self.streamable_budget = [sb.get_weight_streaming_budget() for sb in ws_budget]
        self.ws_budget = ws_budget
        self.module = module

    def __enter__(self) -> "_WeightStreamingContextManager":
        return self

    def __exit__(self, *args: Any) -> None:
        self._reset_context()

    def _reset_context(self):
        for ws_budget in self.ws_budget:
            ws_budget.reset_context()

    def get_streamable_weight_bytes(self):
        return sum(self.streamable_budget)

    def set_streamable_weight_bytes(self, budget_bytes):
        # The weight streaming budget cannot be modified while there are active IExecutionContexts
        self._reset_context()

        total_bytes = self.get_streamable_weight_bytes()
        if total_bytes <= budget_bytes:
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
        for i, ws_budget in enumerate(self.ws_budget):
            ws_budget.set_weight_streaming_budget(normalized_size[i])
            logger.debug(
                f"Set weight streaming size {normalized_size[i]} for {ws_budget.name}"
            )

        return total_bytes


def enable_weight_streaming(module) -> _WeightStreamingContextManager:
    return _WeightStreamingContextManager(module)
