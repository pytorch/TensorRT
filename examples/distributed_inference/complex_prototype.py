import unittest
from functools import partial
from typing import Any, List, Optional, Sequence, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from complex_pass import ComplexOpDetector
from distributed_utils import initialize_distributed_env
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.utils import detect_fake_mode
from torch._functorch.aot_autograd import aot_export_joint_simple
from torch.distributed._tensor import Shard
from torch.distributed.device_mesh import DeviceMesh

# tensor parallel imports
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from torch.fx import GraphModule, Node

device_mesh, _world_size, _rank, logger = initialize_distributed_env(
    "./complex_prototype"
)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.freqs_cis = None  # To be set during forward

    def precompute_freqs(self, end: int, theta: float = 10000.0):
        # freqs_cis: [seq, dim/2, 2]
        freqs = 1.0 / (theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()  # shape [seq, dim/2]
        return torch.polar(torch.ones_like(freqs), freqs)  # complex64
        # unpack into real and imag for TRT compatibility
        # return torch.stack([polar.real, polar.imag], dim=-1)  # shape [seq, dim/2, 2]

    def forward(self, xq: torch.Tensor, xk: torch.Tensor):
        # xq, xk: [batch, seq, heads, dim]
        # freqs_cis: [seq, dim/2, 2] (real + imag)
        # xq shape [batch, seq, heads, dim]
        # xq_ shape [batch, seq, heads, dim/2]

        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

        # broadcast freq to xq_/xk_
        freqs_cis = self.precompute_freqs(xq_.shape[1])
        # freqs_cis = freqs_cis.to(xq_.device)
        freqs_cis = freqs_cis[None, :, None, :]  # [1, seq, 1, dim/2]

        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

        return xq_out.type_as(xq), xk_out.type_as(xk)

##function of rotary embedding##
def rotary_embedding(xq, xk, dim, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(xq.shape[1], device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    freqs_cis = freqs_cis[None, :, None, :]  # [1, seq, 1, dim/2]
    # in our case freqs_cis - 128 * 64 (end * dim/2)
    # after broadcast 1 * 128 * 1 * 64
    # xq_ - 2 * 128 * 4 * 64 (batch * seq * heads * dim/2)
    # in parallel case, xq_ - 2 * 128 * 4 * 32 (batch * seq * heads * dim/2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class RotaryAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.rotary = RotaryEmbedding(dim)
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wo = nn.Linear(dim, dim)

    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        q, k = self.rotary(q, k)
       
        #q, k = rotary_embedding(q, k, self.rotary.dim)
        # normally you'd apply attention here, then project with wo
        return self.wo(q)


########Tensor Parallel########
def parallel_rotary_block(rotary_block, tp_mesh):
    if tp_mesh.size() <= 1:
        logger.info("NO TENSOR PARALLEL, skipping")
        return

    plan = {
        "rotary": SequenceParallel(),
        "wq": ColwiseParallel(),
        "wk": ColwiseParallel(),
        "wo": RowwiseParallel(output_layouts=Shard(0)),
    }

    parallelize_module(rotary_block, tp_mesh, plan)


#######Backend for torch.compile########
########Backend1########
def custom_backendOne(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[Any], **kwargs: Any
):
    print(
        f"The sample_inputs for backendOne: in aot_export_joint_simple {sample_inputs}"
    )
    logger.info(
        f"The sample_inputs for backendOne: in aot_export_joint_simple {sample_inputs}"
    )
    for input_no, input in enumerate(sample_inputs):
        print(f"BackendOne input {input_no} dtype: {input.dtype} shape: {input.shape}")
        logger.info(f"BackendOne input {input_no} dtype: {input.dtype} shape: {input.shape}")
    fake_mode = detect_fake_mode(sample_inputs)
    with unittest.mock.patch.object(
        fake_mode, "allow_non_fake_inputs", True
    ), fake_mode:
        torch_inputs = [
            input for input in sample_inputs if isinstance(input, torch.Tensor)
        ]
    gm = aot_export_joint_simple(
        gm,
        torch_inputs,
        trace_joint=False,
    )
    return gm


########Backend2 and Backend4########
def aux_backend(
    gm: torch.fx.GraphModule,
    sample_inputs: Sequence[Any],
    is_complex_detection: bool = False,
):
    print(
        f"The sample_inputs for aux_backend: in aot_autograd in backendTwo and backendFour {sample_inputs}"
    )
    logger.info(
        f"The sample_inputs for aux_backend: in aot_autograd in backendTwo and backendFour {sample_inputs}"
    )
    for input_no, input in enumerate(sample_inputs):
        print(f"aux_backend input {input_no} dtype: {input.dtype} shape: {input.shape}")
        logger.info(f"aux_backend input {input_no} dtype: {input.dtype} shape: {input.shape}")
    fake_mode = detect_fake_mode(sample_inputs)
    with unittest.mock.patch.object(
        fake_mode, "allow_non_fake_inputs", True
    ), fake_mode:
        torch_inputs = [
            input for input in sample_inputs if isinstance(input, torch.Tensor)
        ]
    if is_complex_detection:
        print(f"The gm graph after aot_autograd with complex detection: {gm.graph}")
        logger.info(f"The gm graph after aot_autograd with complex detection: {gm.graph}")

        complex_op_detector = ComplexOpDetector()
        complex_subgraphs = complex_op_detector.find_complex_op_subgraphs(
            gm, anchor_target=torch.ops.aten.view_as_real.default
        )
        for complex_subgraph in complex_subgraphs:
            print(
                f"complex_subgraph in aot_autograd with complex detection===== {complex_subgraph}"
            )
            logger.info(
                f"complex_subgraph in aot_autograd with complex detection===== {complex_subgraph}"
            )
        return gm
    else:
        print(f"The gm graph after aot_autograd and no complex detection: {gm.graph}")
        logger.info(
            f"The gm graph after aot_autograd and no complex detection: {gm.graph}"
        )
        return gm


########Backend3########
######This backend is with aot_export_joint_simple, with detection of complex inputs and complex graph########
def custom_backendThree(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[Any], **kwargs: Any
):
    print(
        f"The sample_inputs for custom_backendThree: in aot_export_joint_simple {sample_inputs}"
    )
    logger.info(
        f"The sample_inputs for custom_backendThree: in aot_export_joint_simple {sample_inputs}"
    )
    for input_no, input in enumerate(sample_inputs):
        print(f"BackendThree input {input_no} dtype: {input.dtype} shape: {input.shape}")
        logger.info(f"BackendThree input {input_no} dtype: {input.dtype} shape: {input.shape}")
    fake_mode = detect_fake_mode(sample_inputs)
    with unittest.mock.patch.object(
        fake_mode, "allow_non_fake_inputs", True
    ), fake_mode:
        torch_inputs = [
            input for input in sample_inputs if isinstance(input, torch.Tensor)
        ]
    print(f"The gm graph before aot_export_joint_simple: {gm.graph}")
    logger.info(f"The gm graph before aot_export_joint_simple: {gm.graph}")
    gm = aot_export_joint_simple(
        gm,
        torch_inputs,
        trace_joint=False,
    )
    print(f"The gm graph after aot_export_joint_simple: {gm.graph}")
    logger.info(f"The gm graph after aot_export_joint_simple: {gm.graph}")
    complex_op_detector = ComplexOpDetector()
    print("Find subgraphs with complex ops....")
    logger.info("Find subgraphs with complex ops....")
    complex_subgraphs = complex_op_detector.find_complex_op_subgraphs(
        gm, anchor_target=torch.ops.aten.view_as_real.default
    )
    for complex_subgraph in complex_subgraphs:
        print(f"complex_subgraph in aot_export_joint_simple====={complex_subgraph}")
        logger.info(
            f"complex_subgraph in aot_export_joint_simple====={complex_subgraph}"
        )
    return gm


########Backend4########
######This backend is with aot_auto with detection of complex inputs and complex graph########
######Uses auxbackend############
def custom_backendTwoandFour(backend_compiler, is_complex_detection: bool = False):
    if is_complex_detection:
        backend_compiler_parallel = partial(backend_compiler, is_complex_detection=True)
        gm = aot_autograd(fw_compiler=backend_compiler_parallel, bw_compiler=None)
    else:
        gm = aot_autograd(fw_compiler=backend_compiler, bw_compiler=None)
        print(
            f"Test gm graph after aot_autograd and no complex detection in custom_backendTwoandFour"
        )
        logger.info(
            f"Test gm graph after aot_autograd and no complex detection in custom_backendTwoandFour"
        )
    return gm


def main():
    BATCH = 2
    SEQ_LEN = 128
    HEADS = 4
    DIM = 128

    #########Torch.compile for Rotary Embedding########
    xq = torch.randn(BATCH, SEQ_LEN, HEADS, DIM)
    xk = torch.randn(BATCH, SEQ_LEN, HEADS, DIM)
    model = RotaryEmbedding(DIM)
    compiled_model_backendOne = torch.compile(
        model,
        backend=custom_backendOne,
    )
    xq_out, xk_out = compiled_model_backendOne(xq, xk)

    print(f"BackendOne: Non parallelized xq_out.shape: {xq_out.shape}")
    print(f"BackendOne: Non parallelized xk_out.shape: {xk_out.shape}")
    logger.info(f"BackendOne: Non parallelized xq_out.shape: {xq_out.shape}")
    logger.info(f"BackendOne: Non parallelized xk_out.shape: {xk_out.shape}")
    compiled_model_backendTwo = torch.compile(
        model,
        backend=custom_backendTwoandFour(aux_backend),
    )
    xq_out, xk_out = compiled_model_backendTwo(xq, xk)

    print(f"BackendTwo: Non parallelized xq_out.shape: {xq_out.shape}")
    print(f"BackendTwo: Non parallelized xk_out.shape: {xk_out.shape}")
    logger.info(f"BackendTwo: Non parallelized xq_out.shape: {xq_out.shape}")
    logger.info(f"BackendTwo: Non parallelized xk_out.shape: {xk_out.shape}")

    #####Rotarty attention with no tenosor parallel######
    model = RotaryAttention(DIM)
    x = torch.randn(BATCH, SEQ_LEN, HEADS, DIM)
    logger.info("START TESTING WITH ROTARY ATTENTION")
    #########Backend2########
    compiled_parallel_model_backendTwo = torch.compile(
        model,
        backend=custom_backendTwoandFour(aux_backend),
    )
    x_out = compiled_parallel_model_backendTwo(x)
    print(f"BackendTwo: Non parallel rotary attention model x_out.shape: {x_out.shape}")
    logger.info(f"BackendTwo: Non parallel rotary attention model x_out.shape: {x_out.shape}")

    #########Backend4########
    logger.info("START TESTING WITH BACKEND FOUR")
    compiled_parallel_model_backendFour = torch.compile(
        model,
        backend=custom_backendTwoandFour(aux_backend, is_complex_detection=True),
    )
    x_out = compiled_parallel_model_backendFour(x)
    print(f"BackendFour: Non parallel rotary attention model x_out.shape: {x_out.shape}")
    logger.info(f"BackendFour: Non parallel rotary attention model x_out.shape: {x_out.shape}")

    compiled_parallel_model_backendOne = torch.compile(
        model,
        backend=custom_backendOne,
    )
    x_out = compiled_parallel_model_backendOne(x)
    print(f"BackendOne: Non parallel rotary attention model x_out.shape: {x_out.shape}")
    logger.info(f"BackendOne: Non parallel rotary attention model x_out.shape: {x_out.shape}")

    compiled_parallel_model_backendThree = torch.compile(
        model,
        backend=custom_backendThree,
    )
    x_out = compiled_parallel_model_backendThree(x)
    print(f"BackendThree: Non parallel rotary attention model x_out.shape: {x_out.shape}")
    logger.info(f"BackendThree: Non parallel rotary attention model x_out.shape: {x_out.shape}")

    ########################TENSOR PARALLEL########################
    #########Tensor Parallel for Rotary Attention########
    # commented this now for my local setup since only single GPU and I use mpirun in this case
    # device_mesh = DeviceMesh("cuda", torch.arange(torch.cuda.device_count()))

    model = RotaryAttention(DIM)
    parallel_rotary_block(model, device_mesh)
    x = torch.randn(BATCH, SEQ_LEN, HEADS, DIM)

    logger.info("START TESTING WITH TENSOR PARALLEL")

    #########Backend2########
    compiled_parallel_model_backendTwo = torch.compile(
        model,
        backend=custom_backendTwoandFour(aux_backend),
    )
    logger.info("after compilation")
    x_out = compiled_parallel_model_backendTwo(x)
    print(f"BackendTwo: parallel model x_out.shape: {x_out.shape}")
    logger.info(f"BackendTwo: parallel model x_out.shape: {x_out.shape}")

    #########Backend4########
    logger.info("START TESTING WITH BACKEND FOUR")
    compiled_parallel_model_backendFour = torch.compile(
        model,
        backend=custom_backendTwoandFour(aux_backend, is_complex_detection=True),
    )
    x_out = compiled_parallel_model_backendFour(x)
    print(f"BackendFour: parallel model x_out.shape: {x_out.shape}")
    logger.info(f"BackendFour: parallel model x_out.shape: {x_out.shape}")


    #########Backend1########
    # As expected this backend fails in multiple GPUs but works in single GPU
    # Error in multiple GPUs:
    # [rank0]:     raise RuntimeError(
    # [rank0]: torch._dynamo.exc.BackendCompilerFailed: backend='custom_backendOne' raised:
    # [rank0]: RuntimeError: aot_export is not currently supported with traceable tensor subclass.
    # [rank0]: If you need this feature, please comment on <CREATE_ISSUE_LINK>
    compiled_parallel_model_backendOne = torch.compile(
        model,
        backend=custom_backendOne,
    )
    x_out = compiled_parallel_model_backendOne(x)
    print(f"BackendOne: parallel model x_out.shape: {x_out.shape}")
    logger.info(f"BackendOne: parallel model x_out.shape: {x_out.shape}")

    #########Backend3########
    # As expected this backend fails in multiple GPUs but works in single GPU
    # Error in multiple GPUs:
    # [rank0]:     raise RuntimeError(
    # [rank0]: torch._dynamo.exc.BackendCompilerFailed: backend='custom_backendOne' raised:
    # [rank0]: RuntimeError: aot_export is not currently supported with traceable tensor subclass.
    # [rank0]: If you need this feature, please comment on <CREATE_ISSUE_LINK>
    compiled_parallel_model_backendThree = torch.compile(
        model,
        backend=custom_backendThree,
    )
    x_out = compiled_parallel_model_backendThree(x)
    print(f"BackendThree: parallel model x_out.shape: {x_out.shape}")
    logger.info(f"BackendThree: parallel model x_out.shape: {x_out.shape}")


if __name__ == "__main__":
    main()
