# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import gc
import io
import tempfile
import unittest
from pathlib import Path

import torch
import torch_tensorrt as torchtrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import TorchTensorRTModule
from torch_tensorrt.runtime import (
    RuntimeSettings,
    apply_runtime_settings,
    runtime_cache,
)


class ConvReluModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x):
        return torch.relu(self.conv(x))


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Cross-feature runtime tests require TensorRT-RTX",
)
class TestTRTRTXCrossFeature(TestCase):
    def setUp(self):
        super().setUp()
        self.addCleanup(torch._dynamo.reset)
        self.addCleanup(torchtrt.runtime.set_cudagraphs_mode, False)
        torchtrt.runtime.set_cudagraphs_mode(False)
        torch.manual_seed(0)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache_path = str(Path(self.directory.name) / "cache.bin")

    def _compile(self, model, settings, *, dynamic=False):
        if dynamic:
            spec = torchtrt.Input(
                min_shape=(1, 3, 16, 16),
                opt_shape=(2, 3, 16, 16),
                max_shape=(4, 3, 16, 16),
                dtype=torch.float32,
            )
        else:
            spec = torchtrt.Input((2, 3, 16, 16), dtype=torch.float32)
        compiled = torchtrt.compile(
            model,
            ir="dynamo",
            inputs=[spec],
            enabled_precisions={torch.float32},
            min_block_size=1,
            require_full_compilation=True,
        )
        self.assertTrue(
            any(
                isinstance(module, TorchTensorRTModule) for module in compiled.modules()
            )
        )
        apply_runtime_settings(compiled, settings)
        return compiled

    def _check_batches(self, compiled, model, batches):
        with torch.no_grad():
            for iteration, batch in enumerate(batches):
                with self.subTest(iteration=iteration, batch=batch):
                    x = torch.randn(batch, 3, 16, 16, device="cuda")
                    torch.testing.assert_close(
                        compiled(x).clone(), model(x), rtol=1e-3, atol=1e-3
                    )

    def test_cache_persists_with_cuda_graphs(self):
        model = ConvReluModel().eval().cuda()
        compiled = self._compile(
            model,
            RuntimeSettings(
                runtime_cache=self.cache_path, cuda_graph_strategy="whole_graph_capture"
            ),
        )
        self._check_batches(compiled, model, (2, 2, 2))
        del compiled
        gc.collect()
        self.assertTrue(Path(self.cache_path).is_file())
        self.assertGreater(Path(self.cache_path).stat().st_size, 0)

    @parameterized.expand([("lazy",), ("eager",)])
    def test_cache_reuse_across_dynamic_shapes(self, strategy):
        model = ConvReluModel().eval().cuda()
        settings = RuntimeSettings(
            runtime_cache=None, dynamic_shapes_kernel_specialization_strategy=strategy
        )
        compiled = self._compile(model, settings, dynamic=True)
        with runtime_cache(compiled, self.cache_path) as cache:
            self._check_batches(compiled, model, (2, 2))
        saved = Path(self.cache_path).read_bytes()
        self.assertGreater(len(saved), 0)
        del compiled, cache
        gc.collect()

        compiled = self._compile(model, settings, dynamic=True)
        with runtime_cache(compiled, self.cache_path) as cache:
            loaded = io.BytesIO()
            cache.save_to_stream(loaded)
            self.assertTrue(loaded.getvalue() == saved, "The cache did not load.")
            self._check_batches(compiled, model, (2, 4, 1, 2))
            self.assertTrue(cache.has_cache())

    @parameterized.expand([("lazy",), ("eager",)])
    def test_dynamic_shapes_with_cuda_graphs(self, strategy):
        model = ConvReluModel().eval().cuda()
        compiled = self._compile(
            model,
            RuntimeSettings(
                runtime_cache=None,
                dynamic_shapes_kernel_specialization_strategy=strategy,
                cuda_graph_strategy="whole_graph_capture",
            ),
            dynamic=True,
        )
        self._check_batches(compiled, model, (1, 1, 2, 2, 4, 4, 2, 2))


if __name__ == "__main__":
    run_tests()
