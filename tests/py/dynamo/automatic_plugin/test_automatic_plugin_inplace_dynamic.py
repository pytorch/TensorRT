"""Aliased plugin I/O combined with dynamic shapes — the production case for
KV-cache-style ops where the cache tensor's batch dim varies."""

import platform
import unittest

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests

import torch_tensorrt


@torch.library.custom_op(
    "torchtrt_ex::add_one_inplace_dyn", mutates_args=("X",)
)  # type: ignore[misc]
def add_one_inplace_dyn(X: torch.Tensor) -> torch.Tensor:
    assert X.is_cuda
    X.add_(1)
    return X.clone()


@torch.library.register_fake("torchtrt_ex::add_one_inplace_dyn")
def _(X: torch.Tensor) -> torch.Tensor:
    return X


if torch_tensorrt.ENABLED_FEATURES.qdp_plugin:
    torch_tensorrt.dynamo.conversion.plugins.custom_op(
        "torchtrt_ex::add_one_inplace_dyn", supports_dynamic_shapes=True
    )


@unittest.skipIf(
    platform.system() == "Windows",
    "QDP in-place test requires Linux",
)
@unittest.skipIf(
    not torch_tensorrt.ENABLED_FEATURES.qdp_plugin,
    "QDP Plugin is not available",
)
class TestInplacePluginDynamicShapes(unittest.TestCase):
    def test_dynamic_batch(self):
        class Model(nn.Module):
            def forward(self, x):
                return torch.ops.torchtrt_ex.add_one_inplace_dyn.default(x)

        compile_input = torch.randn(8, 32, device="cuda", dtype=torch.float)
        compiled = torch_tensorrt.compile(
            Model(),
            inputs=[
                torch_tensorrt.Input(
                    min_shape=(1, 32),
                    opt_shape=(8, 32),
                    max_shape=(16, 32),
                    dtype=torch.float,
                )
            ],
            ir="dynamo",
            min_block_size=1,
            immutable_weights=True,
        )

        from torch_tensorrt.dynamo.runtime import (
            PythonTorchTensorRTModule,
            TorchTensorRTModule,
        )

        engine_submodules = [
            m
            for _, m in compiled.named_modules()
            if isinstance(m, (PythonTorchTensorRTModule, TorchTensorRTModule))
        ]
        self.assertGreaterEqual(
            len(engine_submodules),
            1,
            f"Expected at least one TRT engine submodule, got graph:\n{compiled.graph}",
        )

        for batch in (1, 4, 16):
            base = torch.randn(batch, 32, device="cuda", dtype=torch.float)
            expected = base + 1
            trt_input = base.clone()
            trt_out = compiled(trt_input)
            torch.testing.assert_close(trt_out, expected, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(trt_input, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    run_tests()
