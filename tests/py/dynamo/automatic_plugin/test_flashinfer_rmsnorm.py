import flashinfer

import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from ..conversion.harness import DispatchTestCase
import flashinfer


@torch.library.custom_op("torchtrt_ex::flashinfer_rmsnorm", mutates_args=())  # type: ignore[misc]
def flashinfer_rmsnorm(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    return flashinfer.norm.rmsnorm(input, weight)


@torch.library.register_fake("torchtrt_ex::flashinfer_rmsnorm")
def _(input: torch.Tensor, weight: torch.Tensor, b: float = 1e-6) -> torch.Tensor:
    return input



torch_tensorrt.dynamo.conversion.plugins.custom_op(
    "torchtrt_ex::flashinfer_rmsnorm", supports_dynamic_shapes=True
)


class TestAutomaticPlugin(DispatchTestCase):
    @parameterized.expand(
        [
            ((64, 64), (64, ), torch.float16),
            ((256, 256), (256, ), torch.float16),
        ]
    )
    def test_rmsnorm_float(self, input_shape, weight_shape, dtype):
        class rmsnorm(nn.Module):
            def forward(self, input, weight):
                return torch.ops.torchtrt_ex.flashinfer_rmsnorm.default(input, weight)

        inputs = [torch.randn(input_shape, device="cuda", dtype=dtype),  torch.randn(weight_shape, device="cuda", dtype=dtype)]

        self.run_test(rmsnorm(), inputs)


if __name__ == "__main__":
    run_tests()