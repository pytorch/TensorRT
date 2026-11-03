# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import unittest
from unittest import mock

import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes import complex_decomposition_adapter
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

    @unittest.skipIf(
        not torch.cuda.is_available()
        or not complex_decomposition_adapter.has_complex_decomposition(),
        "needs CUDA and the upstream complex decomposition",
    )
    def test_offloaded_constants_reach_the_complex_decomposition(self):
        # offload_module_to_cpu leaves folded constants on the CPU; the complex
        # decomposition retraces the graph and must not fall back over them.
        class Rotary(torch.nn.Module):
            def __init__(self):
                super().__init__()
                angles = torch.outer(torch.arange(16.0), torch.ones(4))
                self.register_buffer(
                    "freqs", torch.polar(torch.ones_like(angles), angles)
                )
                # Read only by constant folding, so the retraced module drops it.
                self.scale = torch.nn.Parameter(torch.ones(4))

            def forward(self, x):
                freqs = self.freqs[: x.shape[0]] * self.scale.mean()
                z = torch.view_as_complex(x.reshape(x.shape[0], 4, 2))
                return torch.view_as_real(z * freqs).flatten(1)

        model = Rotary().eval().cuda()
        example_input = torch.randn(8, 8, device="cuda")
        with (
            torch.no_grad(),
            mock.patch.object(
                complex_decomposition_adapter,
                "complex_graph_detection",
                side_effect=AssertionError("fell back to the legacy complex pass"),
            ),
        ):
            expected = model(example_input)
            exported_program = torch.export.export(model, (example_input,))
            compiled_model = torch_tensorrt.dynamo.compile(
                exported_program,
                inputs=[example_input],
                min_block_size=1,
                offload_module_to_cpu=True,
            )
            actual = compiled_model(example_input)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(model.scale.device.type, "cpu")

    @unittest.skipIf(
        not torch.cuda.is_available()
        or not complex_decomposition_adapter.has_complex_decomposition(),
        "needs CUDA and the upstream complex decomposition",
    )
    def test_complex_fallback_does_not_hold_device_copies_of_offloaded_constants(
        self,
    ):
        from torch._functorch._aot_autograd import complex_decomposition

        class Rotary(torch.nn.Module):
            def __init__(self):
                super().__init__()
                angles = torch.outer(torch.arange(256.0), torch.ones(512))
                self.register_buffer(
                    "freqs", torch.polar(torch.ones_like(angles), angles)
                )
                self.scale = torch.nn.Parameter(torch.ones(512))

            def forward(self, x):
                z = torch.view_as_complex(x.reshape(256, 512, 2))
                return torch.view_as_real(z * (self.freqs * self.scale.mean()))

        settings = CompilationSettings(offload_module_to_cpu=True)
        example_input = torch.randn(256, 1024, device="cuda")
        exported_program = torch.export.export(Rotary().cuda(), (example_input,))
        # Folds freqs * scale into a 1 MiB constant kept on the CPU.
        gm = constant_fold(exported_program.module(), settings)

        def failing_retrace(gm, flat_args, decompositions):
            # Like a real retrace, its frame holds the constants it was handed.
            constants = [
                getattr(gm, n.target) for n in gm.graph.nodes if n.op == "get_attr"
            ]
            raise NotImplementedError(f"unsupported op, {len(constants)} constants")

        allocated = {}

        def legacy_pass(gm, settings):
            allocated["fallback"] = torch.cuda.memory_allocated()
            return gm

        with (
            mock.patch.object(
                complex_decomposition, "decompose_complex_in_graph", failing_retrace
            ),
            mock.patch.object(
                complex_decomposition_adapter, "complex_graph_detection", legacy_pass
            ),
        ):
            allocated["before"] = torch.cuda.memory_allocated()
            complex_decomposition_adapter.complex_decomposition_adapter(gm, settings)

        self.assertEqual(allocated["fallback"], allocated["before"])


if __name__ == "__main__":
    run_tests()
