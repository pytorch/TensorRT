import unittest
from math import ceil

import cuda.tile as ct
import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt._enums import dtype

from ...conversion.harness import DispatchTestCase
from .matmul import matmul_kernel


@torch.library.custom_op("cutile::matmul", mutates_args=())  # type: ignore[misc]
def cutile_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    C = torch.empty(A.shape[0], B.shape[1], device=A.device, dtype=A.dtype)
    tm, tn, tk = 256, 256, 64
    m, n, _ = A.shape[0], B.shape[1], A.shape[1]
    grid = (ceil(m / tm) * ceil(n / tn), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, matmul_kernel, (A, B, C, tm, tn, tk))
    return C


@torch.library.register_fake("cutile::matmul")
def _(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.empty(A.shape[0], B.shape[1], device=A.device, dtype=A.dtype)


if not torch_tensorrt.ENABLED_FEATURES.tensorrt_rtx:
    torch_tensorrt.dynamo.conversion.plugins.custom_op(
        "cutile::matmul", supports_dynamic_shapes=True
    )


@unittest.skipIf(
    torch.cuda.get_device_capability() < (10, 0),
    "cuTile requires compute capability 10.0 or later",
)
class TestAutomaticPlugin(DispatchTestCase):

    @parameterized.expand(
        [
            ((64, 64), (64, 128), torch.float16),
            ((256, 256), (256, 16), torch.float16),
        ]
    )
    def test_matmul(self, a_shape, b_shape, data_type):
        class matmul(nn.Module):
            def forward(self, a, b):
                return torch.ops.cutile.matmul.default(a, b)

        inputs = [
            torch.randn(a_shape, device="cuda", dtype=data_type),
            torch.randn(b_shape, device="cuda", dtype=data_type),
        ]

        self.run_test(matmul(), inputs, precision=dtype.f16)


if __name__ == "__main__":
    run_tests()
