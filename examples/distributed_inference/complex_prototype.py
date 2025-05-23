import unittest
from typing import Any, List, Optional, Sequence, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from complex_pass import ComplexOpDetector
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
        # normally you'd apply attention here, then project with wo
        return self.wo(q)


########Tensor Parallel########
def parallel_rotary_block(rotary_block, tp_mesh):
    if tp_mesh.size() <= 1:
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


########Backend2########
def aux_backend(gm: torch.fx.GraphModule, sample_inputs: Sequence[Any], **kwargs: Any):
    fake_mode = detect_fake_mode(sample_inputs)
    with unittest.mock.patch.object(
        fake_mode, "allow_non_fake_inputs", True
    ), fake_mode:
        torch_inputs = [
            input for input in sample_inputs if isinstance(input, torch.Tensor)
        ]
    print("The gm graph after aot_autograd:", gm.graph)
    complex_op_detector = ComplexOpDetector()
    complex_subgraphs = complex_op_detector.find_complex_op_subgraphs(
        gm, anchor_target=torch.ops.aten.view_as_real.default
    )
    for complex_subgraph in complex_subgraphs:
        print("complex_subgraph in aot_autograd=====", complex_subgraph)
    return gm


def custom_backendTwo(backend_compiler):
    gm = aot_autograd(fw_compiler=backend_compiler)
    return gm


########Backend3########
######This backend is with aot_export_joint_simple, with detection of complex inputs and complex graph########
def custom_backendThree(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[Any], **kwargs: Any
):
    fake_mode = detect_fake_mode(sample_inputs)
    with unittest.mock.patch.object(
        fake_mode, "allow_non_fake_inputs", True
    ), fake_mode:
        torch_inputs = [
            input for input in sample_inputs if isinstance(input, torch.Tensor)
        ]
    print("The gm graph before aot_export_joint_simple:", gm.graph)

    gm = aot_export_joint_simple(
        gm,
        torch_inputs,
        trace_joint=False,
    )
    print("The gm graph after aot_export_joint_simple:", gm.graph)
    complex_op_detector = ComplexOpDetector()
    print("Find subgraphs with complex ops....")
    complex_subgraphs = complex_op_detector.find_complex_op_subgraphs(
        gm, anchor_target=torch.ops.aten.view_as_real.default
    )
    for complex_subgraph in complex_subgraphs:
        print("complex_subgraph in aot_export_joint_simple=====", complex_subgraph)
    return gm


########Backend4########
######This backend is with aot_auto with detection of complex inputs and complex graph########
######Uses auxbackend############
def custom_backendFour(backend_compiler):
    gm = aot_autograd(fw_compiler=backend_compiler)
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

    print("xq_out.shape:", xq_out.shape)
    print("xk_out.shape:", xk_out.shape)

    compiled_model_backendTwo = torch.compile(
        model,
        backend=custom_backendTwo(aux_backend),
    )
    xq_out, xk_out = compiled_model_backendTwo(xq, xk)

    print("xq_out.shape:", xq_out.shape)
    print("xk_out.shape:", xk_out.shape)

    #########Tensor Parallel for Rotary Attention########
    # commented this now for my local setup since only single GPU and I use mpirun in this case
    # device_mesh = DeviceMesh("cuda", torch.arange(torch.cuda.device_count()))
    from distributed_utils import initialize_distributed_env

    device_mesh, _world_size, _rank, logger = initialize_distributed_env(
        "./complex_prototype"
    )

    model = RotaryAttention(DIM)
    parallel_rotary_block(model, device_mesh)
    x = torch.randn(BATCH, SEQ_LEN, HEADS, DIM)
    compiled_parallel_model_backendOne = torch.compile(
        model,
        backend=custom_backendOne,
    )
    x_out = compiled_parallel_model_backendOne(x)

    print("parallel model x_out.shape:", x_out.shape)

    compiled_parallel_model_backendTwo = torch.compile(
        model,
        backend=custom_backendTwo(aux_backend),
    )
    x_out = compiled_parallel_model_backendTwo(x)

    print("parallel modelx_out.shape:", x_out.shape)

    compiled_parallel_model_backendThree = torch.compile(
        model,
        backend=custom_backendThree,
    )
    x_out = compiled_parallel_model_backendThree(x)

    print("parallel modelx_out.shape:", x_out.shape)

    compiled_parallel_model_backendFour = torch.compile(
        model,
        backend=custom_backendFour(aux_backend),
    )
    x_out = compiled_parallel_model_backendFour(x)

    print("parallel modelx_out.shape:", x_out.shape)


if __name__ == "__main__":
    main()
