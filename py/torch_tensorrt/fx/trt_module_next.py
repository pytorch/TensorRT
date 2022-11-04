from operator import truediv
from typing import Any, List, Sequence

import torch
from torch_tensorrt import _C
from torch_tensorrt._Device import Device


class TRTModule(torch.nn.Module):
    def __init__(
        self,
        engine_name: str,
        device_info: Device,
        serialized_engine: bytearray,
        input_names: List[str],
        output_names: List[str],
    ):
        super(TRTModule, self).__init__()
        self.engine = torch.classes.tensorrt.Engine(
            [
                _C.rt.ABI_VERSION,
                engine_name,
                device_info._to_serialized_runtime_device(),
                serialized_engine,
                TRTModule._pack_binding_names(input_names),
                TRTModule._pack_binding_names(output_names),
            ]
        )

    def forward(self, *inputs):
        try:
            assert all([i.issubclass(torch.Tensor) for i in inputs])
        except:
            raise RuntimeError("TRTModule expects a flattened list of tensors as input")

        outputs = torch.ops.tensorrt.execute_engine(list(inputs), self.engine)

        if len(outputs) == 1:
            return outputs[0]

        return tuple(outputs)

    def enable_profiling(self, profiling_results_dir: str = None):
        if profiling_results_dir is not None:
            self.engine.profile_path_prefix = profiling_results_dir
        self.engine.enable_profiling()

    def disable_profiling(self):
        self.engine.disable_profiling()

    def get_layer_info(self) -> str:
        return self.engine.get_engine_layer_info()

    def dump_layer_info(self):
        return self.engine.dump_engine_layer_info()

    @staticmethod
    def _pack_binding_names(binding_names: List[str]) -> str:
        return torch.classes.tensorrt.Engine.BINDING_DELIM.join(binding_names)
