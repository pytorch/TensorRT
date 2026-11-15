# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import unittest
from importlib import metadata

import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt._utils import is_tegra_platform, is_tensorrt_version_supported

from packaging.version import Version

from ..testing_utilities import DECIMALS_OF_AGREEMENT


class TestEnableTRTFlags(TestCase):
    def test_toggle_build_args(self):
        class AddSoftmax(torch.nn.Module):
            def forward(self, x):
                x = 3 * x
                y = x + 1
                return torch.softmax(y, 0)

        inputs = [
            torch.rand(
                3,
                5,
                7,
            ).cuda(),
        ]

        fx_graph = torch.fx.symbolic_trace(AddSoftmax())

        # Validate that the results between Torch and Torch-TRT are similar
        # Enable multiple TRT build arguments
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
            disable_tf32=True,
            sparse_weights=True,
            refit=True,
            num_avg_timing_iters=5,
            workspace_size=1 << 10,
            truncate_double=True,
        )

        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            msg=f"AddSoftmax TRT outputs don't match with the original model.",
        )
        torch._dynamo.reset()

    @unittest.skipIf(
        is_tegra_platform() and is_tensorrt_version_supported("10.8"),
        "DLA is not supported on Jetson platform starting TRT 10.8",
    )
    def test_dla_args(self):
        class AddSoftmax(torch.nn.Module):
            def forward(self, x):
                x = 3 * x
                y = x + 1
                return torch.softmax(y, 0)

        inputs = [
            torch.rand(
                3,
                5,
                7,
            ).cuda(),
        ]

        fx_graph = torch.fx.symbolic_trace(AddSoftmax())

        # Validate that the results between Torch and Torch-TRT are similar
        # Enable multiple TRT build arguments
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            device=torch_tensorrt.Device("dla:0", allow_gpu_fallback=True),
            pass_through_build_failures=True,
            dla_sram_size=1048577,
            dla_local_dram_size=1073741825,
            dla_global_dram_size=536870913,
        )

        # DLA is not present on the active machine
        with self.assertRaises(RuntimeError):
            optimized_model(*inputs).detach().cpu()

        torch._dynamo.reset()


class Small(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x * 2 + 1)


class TestUnknownSettings(TestCase):
    def setUp(self):
        self.x = torch.randn(4, 8).cuda()
        self.program = torch.export.export(Small().cuda(), (self.x,))

    def test_dynamo_compile_rejects_unknown_setting(self):
        with self.assertRaisesRegex(
            TypeError,
            r"compile\(\) got an unexpected keyword argument 'min_blok_size'\. "
            r"Did you mean 'min_block_size'\?",
        ):
            torch_tensorrt.dynamo.compile(
                self.program, arg_inputs=[self.x], min_blok_size=1
            )

    def test_compile_rejects_unknown_setting(self):
        with self.assertRaisesRegex(TypeError, "'min_blok_size'"):
            torch_tensorrt.compile(
                Small().cuda(), ir="dynamo", inputs=[self.x], min_blok_size=1
            )

    def test_serialized_engine_rejects_unknown_setting(self):
        with self.assertRaisesRegex(TypeError, "'min_blok_size'"):
            torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
                self.program, arg_inputs=[self.x], min_blok_size=1
            )
        with self.assertRaisesRegex(TypeError, "'min_blok_size'"):
            torch_tensorrt.convert_method_to_trt_engine(
                Small().cuda(), ir="dynamo", inputs=[self.x], min_blok_size=1
            )

    def test_unknown_setting_without_a_close_match(self):
        with self.assertRaises(TypeError) as ctx:
            torch_tensorrt.dynamo.compile(
                self.program, arg_inputs=[self.x], not_a_setting=1
            )
        self.assertNotIn("Did you mean", str(ctx.exception))

    @unittest.skipIf(
        not torch_tensorrt.ENABLED_FEATURES.windows_cross_compile,
        "Cross compilation for Windows is not available",
    )
    def test_cross_compile_for_windows_rejects_unknown_setting(self):
        with self.assertRaisesRegex(TypeError, "'min_blok_size'"):
            torch_tensorrt.dynamo.cross_compile_for_windows(
                self.program, arg_inputs=[self.x], min_blok_size=1
            )

    def test_removed_setting_warns(self):
        with self.assertWarnsRegex(UserWarning, "`enabled_precisions` was removed"):
            trt_gm = torch_tensorrt.dynamo.compile(
                self.program,
                arg_inputs=[self.x],
                min_block_size=1,
                enabled_precisions={torch.float16},
            )
        torch.testing.assert_close(trt_gm(self.x), Small()(self.x))
        with self.assertWarnsRegex(UserWarning, "`enabled_precisions` was removed"):
            torch_tensorrt.compile(
                Small().cuda(),
                ir="dynamo",
                inputs=[self.x],
                min_block_size=1,
                enabled_precisions={torch.float16},
            )

    def test_settings_shared_with_trace_accept_ir(self):
        # One dict often feeds torch_tensorrt.dynamo.trace and compile, with "ir" in it.
        spec = {"inputs": [self.x], "ir": "dynamo", "min_block_size": 1}
        program = torch_tensorrt.dynamo.trace(Small().cuda(), **spec)
        trt_gm = torch_tensorrt.dynamo.compile(program, **spec)
        torch.testing.assert_close(trt_gm(self.x), Small()(self.x))

    def test_deprecated_setting_and_trace_setting_still_accepted(self):
        with self.assertWarns(DeprecationWarning):
            trt_gm = torch_tensorrt.compile(
                Small().cuda(),
                ir="dynamo",
                inputs=[self.x],
                min_block_size=1,
                truncate_long_and_double=True,
                strict=False,
            )
        self.assertTrue(
            any("_run_on_acc" in name for name, _ in trt_gm.named_children())
        )


if __name__ == "__main__":
    run_tests()
