import platform
import unittest

import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

import torch_tensorrt


@torch.library.custom_op("torchtrt_ex::add_one_inplace", mutates_args=("X",))  # type: ignore[misc]
def add_one_inplace(X: torch.Tensor) -> torch.Tensor:
    assert X.is_cuda
    X.add_(1)
    return X.clone()


@torch.library.register_fake("torchtrt_ex::add_one_inplace")
def _(X: torch.Tensor) -> torch.Tensor:
    return X


if torch_tensorrt.ENABLED_FEATURES.qdp_plugin:
    torch_tensorrt.dynamo.conversion.plugins.custom_op(
        "torchtrt_ex::add_one_inplace", supports_dynamic_shapes=False
    )


@unittest.skipIf(
    platform.system() == "Windows",
    "QDP in-place test requires Linux",
)
@unittest.skipIf(
    not torch_tensorrt.ENABLED_FEATURES.qdp_plugin,
    "QDP Plugin is not available",
)
class TestInplacePlugin(unittest.TestCase):
    """In-place ops mutate their input, so DispatchTestCase.run_test (which
    feeds the same tensor to eager and TRT) double-applies the mutation. We
    use cloned inputs and check both the return value and the in-place write.
    """

    @parameterized.expand(
        [
            ((64, 64), torch.float),
            ((128, 32), torch.float),
        ]
    )
    def test_add_one_inplace(self, input_shape, dtype):
        class Model(nn.Module):
            def forward(self, x):
                return torch.ops.torchtrt_ex.add_one_inplace.default(x)

        base = torch.randn(input_shape, device="cuda", dtype=dtype)

        eager_input = base.clone()
        eager_out = Model()(eager_input)
        expected_post = base + 1
        torch.testing.assert_close(eager_input, expected_post, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(eager_out, expected_post, rtol=1e-5, atol=1e-5)

        trt_input = base.clone()
        compiled = torch_tensorrt.compile(
            Model(),
            inputs=[trt_input.clone()],
            ir="dynamo",
            min_block_size=1,
            immutable_weights=True,
        )

        # Guard against a silent fallback to pure-PyTorch: the eager op
        # already mutates the input, so output-only checks pass even when no
        # TRT engine was built.
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

        trt_out = compiled(trt_input)
        torch.testing.assert_close(trt_out, expected_post, rtol=1e-5, atol=1e-5)
        # Aliased plugin I/O is only active if the engine mutated trt_input;
        # a fresh-output engine would leave it at its pre-call values.
        torch.testing.assert_close(trt_input, expected_post, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    run_tests()
