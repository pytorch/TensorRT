"""Multi-input partial-mutation + multi-output coverage.

Exercises the un-functionalize pass's multi-output branch, the alias-map
build in ``_generate_plugin._generic_plugin_desc`` for ops where only one
input is mutated, and the JIT impl's aliased-output ``copy_`` filter.

Only the *fresh* output is returned by the model. The aliased output is
unused. This is deliberate: TRT's preview-feature ``ALIASED_PLUGIN_IO_10_03``
inserts a defensive copy that breaks aliasing when a multi-output plugin's
aliased output is consumed by another TRT layer in the same engine. The
correctness-critical path the test covers is the multi-output plumbing
itself; coverage for "aliased output consumed downstream" is provided by
the single-output test (which TRT handles correctly).
"""

import platform
import unittest
from typing import Tuple

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests

import torch_tensorrt


@torch.library.custom_op(
    "torchtrt_ex::add_inplace_two_out", mutates_args=("X",)
)  # type: ignore[misc]
def add_inplace_two_out(
    X: torch.Tensor, Y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert X.is_cuda and Y.is_cuda
    X.add_(Y)
    return X.clone(), X * 2


@torch.library.register_fake("torchtrt_ex::add_inplace_two_out")
def _(X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return X, torch.empty_like(X)


if torch_tensorrt.ENABLED_FEATURES.qdp_plugin:
    torch_tensorrt.dynamo.conversion.plugins.custom_op(
        "torchtrt_ex::add_inplace_two_out", supports_dynamic_shapes=False
    )


@unittest.skipIf(
    platform.system() == "Windows",
    "QDP in-place test requires Linux",
)
@unittest.skipIf(
    not torch_tensorrt.ENABLED_FEATURES.qdp_plugin,
    "QDP Plugin is not available",
)
class TestMultiOutputInplacePlugin(unittest.TestCase):
    def test_partial_mutation_fresh_output(self):
        class Model(nn.Module):
            def forward(self, x, y):
                _, b = torch.ops.torchtrt_ex.add_inplace_two_out.default(x, y)
                return b

        x_base = torch.randn(64, 64, device="cuda", dtype=torch.float)
        y_base = torch.randn(64, 64, device="cuda", dtype=torch.float)

        x_eager = x_base.clone()
        _, eager_b = add_inplace_two_out(x_eager, y_base.clone())
        expected_x = x_base + y_base
        expected_b = expected_x * 2
        torch.testing.assert_close(x_eager, expected_x, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(eager_b, expected_b, rtol=1e-5, atol=1e-5)

        x_trt = x_base.clone()
        compiled = torch_tensorrt.compile(
            Model(),
            inputs=[x_trt.clone(), y_base.clone()],
            ir="dynamo",
            min_block_size=1,
            immutable_weights=True,
        )

        from torch_tensorrt.dynamo.runtime import (
            TorchTensorRTModule,
        )

        engine_submodules = [
            m for _, m in compiled.named_modules() if isinstance(m, TorchTensorRTModule)
        ]
        self.assertGreaterEqual(
            len(engine_submodules),
            1,
            f"Expected at least one TRT engine submodule, got graph:\n{compiled.graph}",
        )

        result = compiled(x_trt, y_base.clone())
        torch.testing.assert_close(result, expected_b, rtol=1e-5, atol=1e-5)
        # X was mutated in place; Y was not.
        torch.testing.assert_close(x_trt, expected_x, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    run_tests()
