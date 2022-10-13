from operator import truediv
from typing import Any, List, Sequence

import torch

from torch.classes.tensorrt import Engine
from torch.ops.tensorrt import execute_engine

from torch_tensorrt import (_C, Device)

class TRTModule(torch.nn.module):
    def __init__(
        self,
        engine_name: str,
        device_info: Device,
        serialized_engine: bytearray,
    ):
        super(TRTModule, self).__init__()
        self.engine = Engine([
            _C.rt.ABI_VERSION,
            engine_name,
            device_info._to_internal_cuda_device_str(),
            serialized_engine
        ])

    def forward(self, *inputs):
        try:
            assert all([i.issubclass(torch.Tensor) for i in inputs])
        except:
            raise RuntimeError("TRTModule expects a flattened list of tensors as input")
        outputs = execute_engine(list(inputs), self.engine)
        return tuple(outputs)

    def enable_profiling(self, profiler: None):
        #TODO: CHANGE THIS SO IT MAKE MORE SENSE
        self.engine.debug = True

    def disable_profiling(self):
        #TODO: HERE TOO
        self.engine.debug = False

    def get_layer_info(self) -> str:
        raise RuntimeError("Engine Inspector needs to be implemented")
        #assert TRT VERSION > 8.2
        return self.engine.get_engine_information(_C.LayerInformationFormat.JSON)