import torch
import traceback
import torch._dynamo as td

from typing import Dict

from torch_tensorrt.fx.fx2trt import (
    InputTensorSpec,
    TRTInterpreter,
)
from torch._dynamo.backends.common import fake_tensor_unsupported
import tensorrt as trt
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport
from torch_tensorrt.fx.converter_registry import CONVERTERS

from torch_tensorrt.fx.trt_module import TRTModule
from torch_tensorrt.fx.utils import LowerPrecision

from torch._functorch.aot_autograd import aot_module_simplified, make_boxed_compiler
from torch._decomp import register_decomposition, core_aten_decompositions


DECOMPOSITIONS = {**core_aten_decompositions()}
MAX_NUM_TRT_ENGINES = 10

aten = torch.ops.aten


def replace_inplace_op(aten_op, outplace_op):
    """Replace inplace operation with functional equivalent
    Adapted from:
    https://github.com/pytorch/pytorch/blob/3344d79e3f732dadd5c85b99a7aa1a022f187929/torch/_decomp/decompositions.py#L3355-L3361
    """

    @register_decomposition(aten_op, registry=DECOMPOSITIONS)
    def inplace_op(*args, **kwargs):
        out = outplace_op(*args, **kwargs)
        return args[0].copy_(out)

    return inplace_op


replace_inplace_op(aten.add_, aten.add)
replace_inplace_op(aten.addbmm_, aten.addbmm)
replace_inplace_op(aten.addmm_, aten.addmm)
replace_inplace_op(aten.addmv_, aten.addmv)
replace_inplace_op(aten.baddbmm_, aten.baddbmm)
replace_inplace_op(aten.cumprod_, aten.cumprod)
replace_inplace_op(aten.fill_, aten.fill)
replace_inplace_op(aten.gelu_, aten.gelu)
replace_inplace_op(aten.hardsigmoid_, aten.hardsigmoid)
replace_inplace_op(aten.index_put_, aten.index_put)
replace_inplace_op(aten.index_reduce_, aten.index_reduce)
replace_inplace_op(aten.logit_, aten.logit)
replace_inplace_op(aten.relu_, aten.relu)
replace_inplace_op(aten.renorm_, aten.renorm)
replace_inplace_op(aten.round_, aten.round)
replace_inplace_op(aten.scatter_, aten.scatter)
replace_inplace_op(aten.scatter_add_, aten.scatter_add)
replace_inplace_op(aten.scatter_reduce_, aten.scatter_reduce)


class TorchTensorRTOperatorSupport(OperatorSupport):
    """Class to determine whether the aten operators have converters"""

    def __init__(self, support_dict=None):
        super().__init__(support_dict)

        # Initialize sets of supported/unsupported operators
        self.supported_operators = set()
        self.unsupported_operators = set()

    def is_node_supported(
        self, submodules: Dict[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        if node.target in CONVERTERS.keys():
            # If node is a proper computational node, store the operator
            if not node.is_impure():
                node_name = node._pretty_print_target(node.target)
                self.supported_operators.add(node_name)

            return True
        else:
            if not node.is_impure():
                node_name = node._pretty_print_target(node.target)
                self.unsupported_operators.add(node_name)

            return False

    def print_support_overview(self, num_trt_blocks=None):
        if num_trt_blocks is not None:
            print(f"Number of TensorRT-Accelerated Subgraphs: {num_trt_blocks}\n")

        print("Supported Nodes:")
        for node_name in self.supported_operators:
            print(node_name)

        print("\nUnsupported Nodes:")
        for node_name in self.unsupported_operators:
            print(node_name)


def partition(gm: torch.fx.GraphModule, verbose=True):
    """Partition an FX GraphModule with aten ops into TRT engines

    Partitioning is based on operator support
    """
    supported_ops = TorchTensorRTOperatorSupport()
    partitioner = CapabilityBasedPartitioner(gm, supported_ops)

    # Determine partitions, and raise error if the degree of partitioning
    # exceeds a specified threshold
    partitions = partitioner.propose_partitions()
    num_blocks = len(partitions)
    if num_blocks > MAX_NUM_TRT_ENGINES:
        raise AssertionError(
            f"The graph module has {num_blocks} TRT Engines which is larger than the "
            + f"threshold={MAX_NUM_TRT_ENGINES}. Falling back to non-TRT module."
        )

    # Fuse partitions and display overview of supported/unsupported operators
    fused_graph = partitioner.fuse_partitions(partitions)
    num_blocks = len(partitions)

    if verbose:
        supported_ops.print_support_overview(num_blocks)

    return fused_graph


@td.register_backend(name="tensorrt")
@fake_tensor_unsupported
def tensorrt_backend(gm, sample_inputs):
    # Invoke AOTAutograd to translate operators to aten
    return aot_module_simplified(
        gm,
        sample_inputs,
        fw_compiler=make_boxed_compiler(fx2trt_compiler),
        decompositions=DECOMPOSITIONS,
    )


def fx2trt(gm: torch.fx.GraphModule, example_inputs, **kwargs):
    partitioned = partition(gm)

    precision = LowerPrecision.FP32

    def get_submod_inputs(mod, submod, inputs):
        """Helper function to get inputs to submodule"""
        acc_inputs = None

        def get_input(self, inputs):
            nonlocal acc_inputs
            acc_inputs = inputs

        handle = submod.register_forward_pre_hook(get_input)
        mod(*inputs)
        handle.remove()
        return acc_inputs

    for name, _ in partitioned.named_children():
        submod = getattr(partitioned, name)

        # Get submodule inputs
        acc_inputs = get_submod_inputs(partitioned, submod, example_inputs)

        # Create TRT Module from submodule
        interp = TRTInterpreter(
            submod,
            InputTensorSpec.from_tensors(acc_inputs),
            explicit_batch_dimension=True,
            logger_level=trt.Logger.VERBOSE,
        )

        r = interp.run(
            max_workspace_size=20 << 30,
            lower_precision=precision,
            profiling_verbosity=trt.ProfilingVerbosity.VERBOSE,
        )
        trt_mod = TRTModule(*r)

        # Replace FX Module with TRT Module
        setattr(partitioned, name, trt_mod)

    return partitioned


@td.register_backend(name="fx_tensorrt")
@fake_tensor_unsupported
def fx2trt_compiler(gm: torch.fx.GraphModule, example_inputs):
    """Helper function to manage translation of FX module to TRT engines"""
    try:
        trt_compiled = fx2trt(gm, example_inputs)
        return trt_compiled
    except:
        traceback.print_exc()
        print(
            "FX2TRT conversion failed on the subgraph. See trace above. "
            + "Returning GraphModule forward instead."
        )
        return gm.forward
