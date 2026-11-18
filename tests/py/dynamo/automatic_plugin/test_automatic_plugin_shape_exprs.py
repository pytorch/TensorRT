# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Generated plugins whose output shape is a symbolic expression of the
input's dynamic dimensions."""

import unittest

import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests


@torch.library.custom_op("torchtrt_ex::sum_halves", mutates_args=())  # type: ignore[misc]
def sum_halves(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return x[..., :h] + x[..., h : 2 * h]


@torch.library.register_fake("torchtrt_ex::sum_halves")
def _(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], x.shape[-1] // 2))


if torch_tensorrt.ENABLED_FEATURES.qdp_plugin:
    torch_tensorrt.dynamo.conversion.plugins.custom_op(
        "torchtrt_ex::sum_halves", supports_dynamic_shapes=True
    )


@unittest.skipIf(
    not torch_tensorrt.ENABLED_FEATURES.qdp_plugin,
    "QDP Plugin is not available",
)
class TestPluginOutputShapeExprs(TestCase):
    def test_floor_division_of_dynamic_dim(self):
        """An output dim of ``n // 2`` (a sympy ``FloorDiv``) must lower to
        the plugin's ShapeExpr arithmetic for every runtime shape."""

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.ops.torchtrt_ex.sum_halves.default(x)

        compiled = torch_tensorrt.compile(
            Model().cuda(),
            ir="dynamo",
            inputs=[
                torch_tensorrt.Input(
                    min_shape=(2, 4), opt_shape=(2, 10), max_shape=(2, 17)
                )
            ],
            min_block_size=1,
            require_full_compilation=True,
        )
        for n in (4, 10, 17):
            x = torch.randn(2, n, device="cuda")
            out = compiled(x)
            self.assertEqual(out.shape, (2, n // 2))
            torch.testing.assert_close(out, sum_halves(x))


if __name__ == "__main__":
    run_tests()
