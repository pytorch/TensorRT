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
    # torch.library forbids returning an input directly; clone to satisfy the
    # no-alias constraint while still letting the registered fake (which
    # returns X by identity) signal aliasing for the TRT plugin descriptor.
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
    """The standard DispatchTestCase.run_test passes the same input tensor to
    eager and TRT, which double-applies the mutation for in-place ops. Use a
    bespoke flow with cloned inputs and verify both that the output matches the
    expected post-mutation value AND that the input buffer was mutated in
    place."""

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

        # Guard against regressing to PyTorch fallback: if the un-functionalize
        # pass stops restoring the mutating op, the partitioner finds 0
        # supported ops and the "compiled" module is just PyTorch eager — the
        # test would still pass on value because the eager op mutates the
        # input itself, masking the real failure.
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
        # The whole point of aliased plugin I/O is that the input buffer is
        # mutated in place by the engine. If the engine had allocated a fresh
        # output buffer, `trt_input` would still hold the pre-call values.
        torch.testing.assert_close(trt_input, expected_post, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    run_tests()
