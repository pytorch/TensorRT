import time

import tensorrt as trt
import torch
import torch.distributed as dist
import torch.nn as nn
import torch_tensorrt
from tensor_parallel_initialize_dist import initialize_distributed_env
from torch.distributed._tensor import Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

"""
This example covers the rotary embedding and rotary attention case for tensor parallel
"""


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, n_parallel=1
) -> torch.Tensor:
    """Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.
    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.
        n_parallel (int, optional): Number of GPUs for parallel computation. Defaults to 1.
    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim // n_parallel, 2).float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def rotary_embedding(xq, xk, dim, freqs_cis=None):
    """This calculates the rotary embedding for the query and key tensors.
    Args:
        xq (torch.Tensor): Query tensor.
        xk (torch.Tensor): Key tensor.
        dim (int): Dimension of the query and key tensors.
        freqs_cis (torch.Tensor, optional): Precomputed frequency tensor. Defaults to None.
    Returns:
        tuple: Tuple containing the rotated query and key tensors.
    """

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return (xq_out.type_as(xq), xk_out.type_as(xk))


########Tensor Parallel########
def parallel_rotary_block(rotary_block, tp_mesh):
    """Parallel rotary block for tensor parallel
    Args:
        rotary_block: Rotary block to parallelize
        tp_mesh: Tensor parallel mesh
    """
    if tp_mesh.size() <= 1:
        return

    plan = {
        "wq": ColwiseParallel(),
        "wk": ColwiseParallel(),
        "wo": RowwiseParallel(output_layouts=Shard(0)),
    }
    rotary_block.n_parallel = 1  # this is for single GPU, to do remove this hardcode

    parallelize_module(rotary_block, tp_mesh, plan)


class RotaryAttention(nn.Module):
    def __init__(self, dim: int, seq_len: int):
        super().__init__()
        self.dim = dim
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wo = nn.Linear(dim, dim)
        self.seq_len = seq_len
        self.n_parallel = 1
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)
        self.init_weights()

    def _precompute_freqs_cis(self) -> torch.Tensor:
        theta = 10000.0
        return precompute_freqs_cis(self.dim, self.seq_len, theta, self.n_parallel)

    def init_weights(self):
        with torch.device(self.freqs_cis.device):
            self.freqs_cis = self.freqs_cis

    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        freqs_cis = self._precompute_freqs_cis().to(q.device)
        q, k = rotary_embedding(q, k, self.dim, freqs_cis=freqs_cis)
        return self.wo(q)
