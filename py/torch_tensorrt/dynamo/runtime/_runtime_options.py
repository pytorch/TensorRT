from enum import Enum
from torch_tensorrt.dynamo.runtime import TensorRTModule
from torch_tensorrt.fx import TRTModule as PyTensorRTModule


class RuntimeOption(Enum):
    TorchTensorRTRuntime = 1
    PythonRuntime = 2


RUNTIMES = {
    RuntimeOption.TorchTensorRTRuntime: TensorRTModule,
    RuntimeOption.PythonRuntime: PyTensorRTModule,
}
