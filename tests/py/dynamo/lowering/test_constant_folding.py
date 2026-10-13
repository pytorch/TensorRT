# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import unittest

import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.constant_folding import constant_fold


class _Permute(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(1, 512, 513))

    def forward(self, x):
        return x + self.weight.permute(0, 2, 1)


class _Squeeze(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(1, 512, 513))

    def forward(self, x):
        return x + self.weight.squeeze()


class _AsStrided(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(512, 513))

    def forward(self, x):
        return x + torch.as_strided(self.weight, size=(512, 513), stride=(513, 1))


class _Select(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(512, 513))

    def forward(self, x):
        return x + self.weight[0]


class TestConstantFolding(TestCase):
    def _export_and_fold(self, model, example_input):
        exported_program = torch.export.export(model, (example_input,))
        return constant_fold(exported_program.module(), CompilationSettings())

    def _assert_installed(self, gm, op):
        self.assertNotIn(op, [node.target for node in gm.graph.nodes])
        self.assertTrue(
            any(name.startswith("_frozen_param") for name, _ in gm.named_parameters())
        )

    def test_folded_constants_are_deepcopy_safe_parameters(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                return x + self.weight.sin()

        example_input = (torch.randn(4, 4),)

        for offload_module_to_cpu in (False, True):
            with self.subTest(offload_module_to_cpu=offload_module_to_cpu):
                exported_program = torch.export.export(Model(), example_input)
                gm = exported_program.module()
                gm = constant_fold(
                    gm,
                    CompilationSettings(offload_module_to_cpu=offload_module_to_cpu),
                )

                frozen_parameters = [
                    parameter
                    for name, parameter in gm.named_parameters()
                    if name.startswith("_frozen_param")
                ]
                self.assertEqual(len(frozen_parameters), 1)

                frozen_parameter = frozen_parameters[0]
                self.assertIsInstance(frozen_parameter, torch.nn.Parameter)
                self.assertFalse(frozen_parameter.requires_grad)
                self.assertIsNone(frozen_parameter.grad_fn)

                copy.deepcopy(gm)

    def test_only_converter_safe_aliased_ops_skip_install(self):
        permute_gm = self._export_and_fold(_Permute(), torch.randn(1, 513, 512))
        self.assertIn(
            torch.ops.aten.permute.default,
            [node.target for node in permute_gm.graph.nodes],
        )
        self.assertFalse(
            any(
                name.startswith("_frozen_param")
                for name, _ in permute_gm.named_parameters()
            )
        )

        for model, example_input, op in (
            (
                _Squeeze(),
                torch.randn(512, 513),
                torch.ops.aten.squeeze.default,
            ),
            (
                _AsStrided(),
                torch.randn(512, 513),
                torch.ops.aten.as_strided.default,
            ),
            (
                _Select(),
                torch.randn(513),
                torch.ops.aten.select.int,
            ),
        ):
            with self.subTest(op=op):
                self._assert_installed(
                    self._export_and_fold(model, example_input),
                    op,
                )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_aliased_view_models_compile_end_to_end(self):
        for model, example_input in (
            (_Permute(), torch.randn(1, 513, 512)),
            (_Squeeze(), torch.randn(512, 513)),
            (_AsStrided(), torch.randn(512, 513)),
            (_Select(), torch.randn(513)),
        ):
            model = model.eval().cuda()
            example_input = example_input.cuda()

            with self.subTest(model=type(model).__name__):
                with torch.no_grad():
                    expected = model(example_input)
                    exported_program = torch.export.export(model, (example_input,))
                    compiled_model = torch_tensorrt.dynamo.compile(
                        exported_program,
                        inputs=[example_input],
                        min_block_size=1,
                        require_full_compilation=True,
                        offload_module_to_cpu=True,
                    )
                    actual = compiled_model(example_input)

                torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    run_tests()
