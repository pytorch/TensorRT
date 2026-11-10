# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Eager execution of an engine that updates an input buffer in place.

The ``tensorrt::execute_engine`` custom op has to clone outputs that alias
inputs, so the Python runtime runs such an engine directly in plain eager
execution: the updated buffer is the caller's tensor, with no copy. Under
tracing, ``torch.func`` transforms, a dispatch mode or autograd the op is
still used.
"""

import unittest

import torch
import torch_tensorrt
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils._python_dispatch import TorchDispatchMode
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
    _python_engine_runs_eagerly,
)


class _InPlaceUpdate(torch.nn.Module):
    def forward(self, buffer, new, position):
        updated = torch.index_copy(buffer, 2, position, new)
        return updated.sum(dim=2), updated


class _PassThrough(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))


@unittest.skipIf(
    ENABLED_FEATURES.torch_tensorrt_runtime,
    "the eager fast path applies to the Python runtime",
)
@unittest.skipIf(not torch.cuda.is_available(), "needs CUDA")
class TestAliasedOutputsInEagerExecution(TestCase):
    def setUp(self):
        buffer = torch.zeros(1, 2, 8, 4, device="cuda")
        new = torch.ones(1, 2, 1, 4, device="cuda")
        position = torch.tensor([3], device="cuda")
        program = torch.export.export(_InPlaceUpdate().cuda(), (buffer, new, position))
        self.compiled = torch_tensorrt.dynamo.compile(
            program, arg_inputs=[buffer, new, position], min_block_size=1
        )
        self.new = new

    def _expected(self, position):
        expected = torch.zeros(1, 2, 8, 4, device="cuda")
        expected[:, :, position] = self.new[:, :, 0]
        return expected

    def test_engine_has_aliased_output(self):
        engines = [
            module
            for module in self.compiled.modules()
            if isinstance(module, torch_tensorrt.dynamo.runtime.TorchTensorRTModule)
        ]
        self.assertEqual(len(engines), 1)
        self.assertTrue(engines[0].aliased_io)

    def test_eager_call_returns_the_updated_buffer_without_a_copy(self):
        buffer = torch.zeros(1, 2, 8, 4, device="cuda")
        position = torch.tensor([5], device="cuda")
        summed, updated = self.compiled(buffer, self.new, position)
        self.assertEqual(updated.data_ptr(), buffer.data_ptr())
        torch.testing.assert_close(buffer, self._expected(5))
        torch.testing.assert_close(summed, self._expected(5).sum(dim=2))

    def test_dispatch_mode_keeps_the_op_and_its_copy(self):
        buffer = torch.zeros(1, 2, 8, 4, device="cuda")
        position = torch.tensor([2], device="cuda")
        with _PassThrough():
            summed, updated = self.compiled(buffer, self.new, position)
        self.assertNotEqual(updated.data_ptr(), buffer.data_ptr())
        torch.testing.assert_close(updated, self._expected(2))
        torch.testing.assert_close(buffer, self._expected(2))
        torch.testing.assert_close(summed, self._expected(2).sum(dim=2))


@unittest.skipIf(
    ENABLED_FEATURES.torch_tensorrt_runtime,
    "the eager fast path applies to the Python runtime",
)
class TestWhenTheEngineRunsOutsideTheOp(TestCase):
    def test_plain_eager_execution_runs_the_engine_directly(self):
        self.assertTrue(_python_engine_runs_eagerly([torch.ones(2)]))
        with torch.inference_mode():
            self.assertTrue(_python_engine_runs_eagerly([torch.ones(2)]))
        with torch.no_grad():
            self.assertTrue(_python_engine_runs_eagerly([torch.ones(2)]))

    def test_tracers_transforms_and_autograd_keep_the_op(self):
        seen = []

        def probe(x):
            seen.append(_python_engine_runs_eagerly([x]))
            return x * 2

        x = torch.ones(2)
        torch.jit.trace(probe, (x,), check_trace=False)
        make_fx(probe, pre_dispatch=True)(x)
        make_fx(probe)(x)
        torch.func.functionalize(probe)(x)
        torch.vmap(probe)(torch.ones(2, 2))
        with _PassThrough():
            probe(x)
        self.assertGreaterEqual(len(seen), 6)
        self.assertFalse(any(seen))
        self.assertFalse(
            _python_engine_runs_eagerly([torch.ones(2, requires_grad=True)])
        )


if __name__ == "__main__":
    run_tests()
