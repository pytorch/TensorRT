import torch
import traceback
import logging
import torch._dynamo as dynamo

from torch_tensorrt.fx.fx2trt import (
    InputTensorSpec,
    TRTInterpreter,
)
import tensorrt as trt
from torch_tensorrt.fx.tools.trt_splitter import (
    TRTSplitter,
    TRTSplitterSetting,
)
from torch_tensorrt.fx.tracer.dispatch_tracer import aten_tracer
from torch_tensorrt.tensorrt_module import TensorRTModule
from torch_tensorrt.dynamo import compile
from torch_tensorrt.fx.utils import LowerPrecision
from py.torch_tensorrt.dynamo.lowering.passes._partition import partition

from torch._dynamo.backends.common import fake_tensor_unsupported

from torch._functorch.aot_autograd import aot_module_simplified, make_boxed_compiler

from torch_tensorrt.dynamo.lowering.decompositions import default_decompositions

MAX_SPLITS_THRESHOLD = 10

log = logging.getLogger(__name__)


@dynamo.register_backend(name="torch_tensorrt")
@fake_tensor_unsupported
def torch_tensorrt_dynamo_backend(gm: torch.fx.GraphModule, example_inputs):
    try:
        trt_compiled = compile(gm, example_inputs)
        return trt_compiled
    except Exception:
        traceback.print_exc()
        print(
            "Torch-TensorRT compilation failed on the subgraph. See trace above. Returning GraphModule forward instead"
        )
        return gm.forward


@dynamo.register_backend(name="aot_torch_tensorrt")
@fake_tensor_unsupported
def aot_torch_tensorrt_dynamo_backend(gm: torch.fx.GraphModule, example_inputs):
    # Invoke AOTAutograd to compile model
    return aot_module_simplified(
        gm,
        example_inputs,
        fw_compiler=make_boxed_compiler(torch_tensorrt_dynamo_backend),
        decompositions=default_decompositions,
    )


# Backends aliased to tensorrt in pytorch/pytorch
