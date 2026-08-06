"""Single-output aliased plugin whose output is consumed by another TRT layer.

This is the realistic production pattern (e.g. a KV-cache update whose
post-update tensor is read by an attention layer in the same engine).
"""

import platform
import unittest

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests

import torch_tensorrt


@torch.library.custom_op(
    "torchtrt_ex::add_one_inplace_consumed", mutates_args=("X",)
)  # type: ignore[misc]
def add_one_inplace_consumed(X: torch.Tensor) -> torch.Tensor:
    assert X.is_cuda
    X.add_(1)
    return X.clone()


@torch.library.register_fake("torchtrt_ex::add_one_inplace_consumed")
def _(X: torch.Tensor) -> torch.Tensor:
    return X


if torch_tensorrt.ENABLED_FEATURES.qdp_plugin:
    torch_tensorrt.dynamo.conversion.plugins.custom_op(
        "torchtrt_ex::add_one_inplace_consumed", supports_dynamic_shapes=False
    )


@unittest.skipIf(
    platform.system() == "Windows",
    "QDP in-place test requires Linux",
)
@unittest.skipIf(
    not torch_tensorrt.ENABLED_FEATURES.qdp_plugin,
    "QDP Plugin is not available",
)
class TestInplacePluginConsumed(unittest.TestCase):
    def test_aliased_output_consumed_downstream(self):
        class Model(nn.Module):
            def forward(self, x):
                a = torch.ops.torchtrt_ex.add_one_inplace_consumed.default(x)
                return a * 2

        x_base = torch.randn(64, 64, device="cuda", dtype=torch.float)
        expected_post = x_base + 1
        expected = expected_post * 2

        x_trt = x_base.clone()
        compiled = torch_tensorrt.compile(
            Model(),
            inputs=[x_trt.clone()],
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

        result = compiled(x_trt)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(x_trt, expected_post, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    run_tests()
