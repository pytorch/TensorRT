# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Single-output aliased plugin whose internal output feeds another TRT layer.

The mutated tensor is created inside the engine, so plugin-level aliasing is
required but no runtime input/output alias record should be emitted.
"""

import platform
import unittest

import torch
import torch.nn as nn
import torch_tensorrt
from torch.testing._internal.common_utils import run_tests


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


@torch.library.custom_op(
    "torchtrt_ex::add_one_and_two", mutates_args=("X", "Y")
)  # type: ignore[misc]
def add_one_and_two(X: torch.Tensor, Y: torch.Tensor) -> None:
    X.add_(1)
    Y.add_(2)


@torch.library.register_fake("torchtrt_ex::add_one_and_two")
def _(X: torch.Tensor, Y: torch.Tensor) -> None:
    pass


if torch_tensorrt.ENABLED_FEATURES.qdp_plugin:
    for op_name in ("add_one_inplace_consumed", "add_one_and_two"):
        torch_tensorrt.dynamo.conversion.plugins.custom_op(
            f"torchtrt_ex::{op_name}", supports_dynamic_shapes=False
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
                internal = x + 2
                a = torch.ops.torchtrt_ex.add_one_inplace_consumed.default(internal)
                return a * 2

        x_base = torch.randn(64, 64, device="cuda", dtype=torch.float)
        expected = (x_base + 3) * 2

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
        self.assertTrue(
            all(not module.aliased_io for module in engine_submodules),
            "Intra-engine plugin aliasing must not create a runtime alias binding",
        )

        result = compiled(x_trt)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(x_trt, x_base, rtol=1e-5, atol=1e-5)

    def test_copy_of_an_input_keeps_the_input(self):
        # Each op makes a new tensor, but TensorRT could keep it in the input's memory:
        # clone hands back its input, and TensorRT drops reshapes, a same-size resize
        # and a pool over one element.
        ops = {
            "clone": (lambda x: x.clone(), (64, 64), torch.float),
            "avg_pool2d": (
                lambda x: nn.functional.avg_pool2d(x, 1),
                (1, 2, 64, 64),
                torch.float,
            ),
            "reshape_clone": (
                lambda x: x.reshape(4096).reshape(64, 64).clone(),
                (64, 64),
                torch.float,
            ),
            "same_size_resize": (
                lambda x: nn.functional.interpolate(x, size=(64, 64)),
                (1, 2, 64, 64),
                torch.float,
            ),
            # TensorRT keeps a concatenation of slices of x in the memory of x.
            "cat_of_slices": (
                lambda x: torch.cat([x[:32], x[32:]]),
                (64, 64),
                torch.float,
            ),
            # TensorRT has no int8 elementwise layer that keeps the copy.
            "clone_int8": (lambda x: x.clone(), (64, 64), torch.int8),
        }
        for name, (op, shape, dtype) in ops.items():
            with self.subTest(name):

                class Model(nn.Module):
                    def forward(self, x):
                        copy = op(x)
                        a = torch.ops.torchtrt_ex.add_one_inplace_consumed.default(copy)
                        return a * 2

                x_base = torch.randn(shape, device="cuda").to(dtype)
                x_trt = x_base.clone()
                compiled = torch_tensorrt.compile(
                    Model(),
                    inputs=[x_trt.clone()],
                    ir="dynamo",
                    min_block_size=1,
                    immutable_weights=True,
                    require_full_compilation=True,
                )
                result = compiled(x_trt)
                torch.testing.assert_close(result, (x_base + 1) * 2)
                torch.testing.assert_close(x_trt, x_base)

    def test_copy_of_a_mutated_input_keeps_the_input(self):
        # The earlier plugins write into x, and the last one must not write into x again.
        for earlier in (1, 2):
            with self.subTest(earlier=earlier):

                class Model(nn.Module):
                    def forward(self, x):
                        for _ in range(earlier):
                            torch.ops.torchtrt_ex.add_one_inplace_consumed.default(x)
                        copy = x.clone()
                        a = torch.ops.torchtrt_ex.add_one_inplace_consumed.default(copy)
                        return a * 2

                x_base = torch.randn(64, 64, device="cuda")
                x_trt = x_base.clone()
                compiled = torch_tensorrt.compile(
                    Model(),
                    inputs=[x_trt.clone()],
                    ir="dynamo",
                    min_block_size=1,
                    immutable_weights=True,
                    require_full_compilation=True,
                )
                result = compiled(x_trt)
                torch.testing.assert_close(result, (x_base + earlier + 1) * 2)
                torch.testing.assert_close(x_trt, x_base + earlier)

    def test_copy_of_a_tensor_read_elsewhere_keeps_it(self):
        # clone hands back its input, so the plugin must not write into y, which the
        # engine also returns.
        copies = {"clone": lambda y: y.clone()}
        for name, copy in copies.items():
            with self.subTest(name):

                class Model(nn.Module):
                    def forward(self, x):
                        y = x + 2
                        a = torch.ops.torchtrt_ex.add_one_inplace_consumed.default(
                            copy(y)
                        )
                        return a * 2, y

                x = torch.randn(64, 64, device="cuda")
                compiled = torch_tensorrt.compile(
                    Model(),
                    inputs=[x.clone()],
                    ir="dynamo",
                    min_block_size=1,
                    immutable_weights=True,
                    require_full_compilation=True,
                )
                torch.testing.assert_close(compiled(x.clone()), Model()(x.clone()))

    def test_copy_of_a_kv_cache_update_keeps_the_cache(self):
        # The KV cache update writes into the memory of the cache input and returns it.
        class Model(nn.Module):
            def forward(self, cache, index, update):
                new = torch.ops.aten.index_copy.default(cache, 2, index, update)
                a = torch.ops.torchtrt_ex.add_one_inplace_consumed.default(new.clone())
                return new, a * 2

        cache = torch.zeros(1, 4, 16, 8, device="cuda")
        index = torch.tensor([3], device="cuda")
        update = torch.full((1, 4, 1, 8), 7.0, device="cuda")
        expected = torch.ops.aten.index_copy.default(cache, 2, index, update)
        compiled = torch_tensorrt.compile(
            Model(),
            inputs=[cache.clone(), index, update],
            ir="dynamo",
            min_block_size=1,
            immutable_weights=True,
            require_full_compilation=True,
        )
        new, a = compiled(cache.clone(), index, update)
        torch.testing.assert_close(new, expected)
        torch.testing.assert_close(a, (expected + 1) * 2)

    def test_input_mutated_through_2_arguments(self):
        # Memory of its own for the clone would split the writes of the 2 arguments, so
        # the plugin writes both into the input, as before.
        class Model(nn.Module):
            def forward(self, x):
                copy = x.clone()
                torch.ops.torchtrt_ex.add_one_and_two.default(copy, copy)
                return copy

        x = torch.ones(8, 8, device="cuda")
        compiled = torch_tensorrt.compile(
            Model(),
            inputs=[x.clone()],
            ir="dynamo",
            min_block_size=1,
            immutable_weights=True,
            require_full_compilation=True,
        )
        torch.testing.assert_close(compiled(x.clone()), x + 3)

    def test_copy_of_a_constant_keeps_the_constant(self):
        # Constant folding turns the clone of the buffer into a constant.
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("w", torch.randn(64, 64))

            def forward(self, x):
                copy = self.w.clone()
                a = torch.ops.torchtrt_ex.add_one_inplace_consumed.default(copy)
                return a * 2 + x

        model = Model().cuda()
        x = torch.zeros(64, 64, device="cuda")
        compiled = torch_tensorrt.compile(
            model,
            inputs=[x],
            ir="dynamo",
            min_block_size=1,
            immutable_weights=True,
            require_full_compilation=True,
        )
        # A second run reads the constant as the first one left it.
        for _ in range(2):
            torch.testing.assert_close(compiled(x), (model.w + 1) * 2)

    @unittest.skipIf(
        torch.cuda.get_device_capability() < (8, 9),
        "FP8 requires compute capability 8.9 or later",
    )
    def test_fp8_copy_builds_as_before(self):
        # TensorRT lets only a few layers, such as a plugin, read FP8 data, so the copy
        # of x is left as it is and the engine builds. The plugin then writes into x.
        class Model(nn.Module):
            def forward(self, x):
                copy = x.clone()
                return torch.ops.torchtrt_ex.add_one_inplace_consumed.default(copy)

        x = torch.zeros(64, 64, device="cuda").to(torch.float8_e4m3fn)
        torch_tensorrt.compile(
            Model(),
            inputs=[x],
            ir="dynamo",
            min_block_size=1,
            immutable_weights=True,
            require_full_compilation=True,
        )

    def test_int8_constant_fails_to_build_as_before(self):
        # Torch-TensorRT stores an int8 constant as int32. With memory of its own, the
        # plugin would compute 127 + 1 in int32 and give 128, where PyTorch gives -128.
        # So the constant is left as it is, and TensorRT does not let a plugin write
        # into it.
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("w", torch.full((4, 4), 127, dtype=torch.int8))

            def forward(self, x):
                copy = self.w.clone()
                a = torch.ops.torchtrt_ex.add_one_inplace_consumed.default(copy)
                return a.to(torch.int32) + x

        x = torch.zeros(4, 4, dtype=torch.int32, device="cuda")
        with self.assertRaisesRegex(RuntimeError, "could not build the engine"):
            torch_tensorrt.compile(
                Model().cuda(),
                inputs=[x],
                ir="dynamo",
                min_block_size=1,
                immutable_weights=True,
                require_full_compilation=True,
            )


if __name__ == "__main__":
    run_tests()
