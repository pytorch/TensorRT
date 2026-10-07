# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for how much host and device memory compilation holds on to."""

import copy
import io
import os
import platform
import unittest
from typing import Any, Callable, List, Tuple
from unittest import mock

import torch
from torch_tensorrt import ENABLED_FEATURES
from torch_tensorrt.dynamo import utils
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule
from torch_tensorrt.logging import TRT_LOGGER

import tensorrt as trt


def _proc_status(field: str) -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024
    raise KeyError(field)


def _peak_rss_growth(fn: Callable[[], Any]) -> Tuple[Any, int]:
    """Run ``fn``; return its result and how far RSS rose above its starting point.

    Uses the kernel's high-water mark (reset through ``/proc/self/clear_refs``), so
    nothing is missed between samples. Freed memory that glibc still holds would absorb
    a copy without raising RSS, hiding it, so that is returned to the OS first.
    """
    malloc_trim = utils._glibc_malloc_trim()
    if malloc_trim is not None:
        malloc_trim(0)
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")
    before = _proc_status("VmRSS")
    result = fn()
    return result, _proc_status("VmHWM") - before


def _build_plan(width: int = 2048, layers: int = 8) -> Tuple[bytes, List[torch.Tensor]]:
    """A plan for ``x @ w.T`` chained over ``layers`` fp16 weights, plus the weights."""
    weights = [
        torch.randn(width, width, dtype=torch.float16) * width**-0.5
        for _ in range(layers)
    ]
    builder = trt.Builder(TRT_LOGGER)
    flags = 0
    if not ENABLED_FEATURES.tensorrt_rtx:
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    net = builder.create_network(flags)
    t = net.add_input("x", trt.float16, (4, width))
    for w in weights:
        c = net.add_constant(
            (width, width), trt.Weights(trt.float16, w.data_ptr(), w.numel())
        )
        t = net.add_matrix_multiply(
            t, trt.MatrixOperation.NONE, c.get_output(0), trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
    t.name = "y"
    net.mark_output(t)
    plan = builder.build_serialized_network(net, builder.create_builder_config())
    return bytes(plan), weights


class TestTrimHostHeap(unittest.TestCase):
    def test_trims_by_default(self):
        malloc_trim = mock.Mock(return_value=1)
        env = {
            k: v
            for k, v in os.environ.items()
            if k != "TORCHTRT_ENABLE_BUILDER_MALLOC_TRIM"
        }
        with mock.patch.object(
            utils, "_glibc_malloc_trim", return_value=malloc_trim
        ), mock.patch.dict(os.environ, env, clear=True):
            utils.trim_host_heap()
        malloc_trim.assert_called_once_with(0)

    def test_env_var_disables_trimming(self):
        malloc_trim = mock.Mock(return_value=1)
        with mock.patch.object(
            utils, "_glibc_malloc_trim", return_value=malloc_trim
        ), mock.patch.dict(os.environ, {"TORCHTRT_ENABLE_BUILDER_MALLOC_TRIM": "0"}):
            utils.trim_host_heap()
        malloc_trim.assert_not_called()

    def test_missing_malloc_trim_is_a_no_op(self):
        with mock.patch.object(utils, "_glibc_malloc_trim", return_value=None):
            utils.trim_host_heap()

    @unittest.skipIf(platform.system() != "Linux", "malloc_trim is glibc-only")
    def test_finds_glibc_malloc_trim(self):
        self.assertIsNotNone(utils._glibc_malloc_trim())
        utils.trim_host_heap()


class TestDeallocateModule(unittest.TestCase):
    def _module(self, n_layers: int, width: int) -> torch.nn.Module:
        m = torch.nn.Sequential(
            *[torch.nn.Linear(width, width) for _ in range(n_layers)]
        ).cuda()
        m.register_buffer("scale", torch.ones(width, device="cuda"))
        return m

    def test_moves_everything_to_cpu_in_place(self):
        m = self._module(n_layers=4, width=256)
        params = list(m.parameters())
        expected = [p.detach().cpu().clone() for p in params]
        utils.deallocate_module(m)
        self.assertTrue(all(t.device.type == "cpu" for t in m.state_dict().values()))
        # Same Parameter objects, so every holder of them (e.g. the ExportedProgram
        # the module came from) sees the CPU copy too, exactly like module.to().
        self.assertTrue(all(a is b for a, b in zip(params, m.parameters())))
        for p, e in zip(m.parameters(), expected):
            torch.testing.assert_close(p.detach(), e)

    def test_moves_sparse_parameters(self):
        m = torch.nn.Module()
        m.dense = torch.nn.Parameter(torch.randn(64, 64, device="cuda"))
        m.sparse = torch.nn.Parameter(torch.randn(64, 64, device="cuda").to_sparse())
        utils.deallocate_module(m, release_every_bytes=1)
        self.assertEqual(m.dense.device.type, "cpu")
        self.assertEqual(m.sparse.device.type, "cpu")
        self.assertTrue(m.sparse.is_sparse)

    def test_releases_gpu_memory_while_moving(self):
        torch.cuda.empty_cache()
        m = self._module(n_layers=8, width=1024)
        layer_bytes = 1024 * 1024 * 4
        before = torch.cuda.memory_allocated()
        reserved_while_moving = []
        real_empty_cache = torch.cuda.empty_cache

        def spy() -> None:
            real_empty_cache()
            reserved_while_moving.append(torch.cuda.memory_reserved())

        with mock.patch.object(torch.cuda, "empty_cache", side_effect=spy):
            utils.deallocate_module(m, release_every_bytes=layer_bytes)

        # Released between tensors, not only once at the end, and the GPU footprint
        # shrinks as the move progresses.
        self.assertGreater(len(reserved_while_moving), 4)
        self.assertLess(reserved_while_moving[-2], reserved_while_moving[0])
        self.assertLessEqual(torch.cuda.memory_allocated(), before - 8 * layer_bytes)


@unittest.skipIf(platform.system() != "Linux", "measures RSS through /proc")
class TestEngineSetupMemory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan, cls.weights = _build_plan()

    def _module(self) -> TorchTensorRTModule:
        return TorchTensorRTModule(
            serialized_engine=self.plan,
            input_binding_names=["x"],
            output_binding_names=["y"],
            name="memory_probe",
            settings=CompilationSettings(),
        )

    def test_setup_does_not_copy_the_plan(self):
        self._module()  # warm up: load runtime libraries and kernels once
        module, growth = _peak_rss_growth(self._module)
        # The engine's weights go to the GPU. Host memory should not hold another copy
        # of the plan; it used to hold two while it was converted for the C++ runtime.
        self.assertLess(growth, len(self.plan) // 2)

        x = torch.randn(4, 2048, dtype=torch.float16, device="cuda")
        expected = x
        for w in self.weights:
            expected = expected @ w.cuda().T
        torch.testing.assert_close(module(x), expected, rtol=2e-2, atol=2e-2)


@unittest.skipIf(
    not ENABLED_FEATURES.torch_tensorrt_runtime,
    "the Python runtime engine keeps its own copy of the plan",
)
class TestPlanRetention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan, cls.weights = _build_plan(width=512, layers=4)
        cls.x = torch.randn(4, 512, dtype=torch.float16, device="cuda")
        expected = cls.x
        for w in cls.weights:
            expected = expected @ w.cuda().T
        cls.expected = expected

    def _module(self, plan: Any = None) -> TorchTensorRTModule:
        return TorchTensorRTModule(
            serialized_engine=self.plan if plan is None else plan,
            input_binding_names=["x"],
            output_binding_names=["y"],
            name="retention_probe",
            settings=CompilationSettings(),
        )

    def _check(self, module: TorchTensorRTModule) -> None:
        torch.testing.assert_close(module(self.x), self.expected, rtol=2e-2, atol=2e-2)

    def test_plan_is_dropped_after_setup(self):
        module = self._module()
        self.assertIsNone(module._serialized_engine)
        # Still readable on demand, and still a working plan.
        plan = module.serialized_engine
        self.assertIsInstance(plan, bytes)
        self._check(self._module(plan))

    def test_lazy_init_keeps_the_plan_until_setup(self):
        module = TorchTensorRTModule(
            serialized_engine=self.plan,
            input_binding_names=["x"],
            output_binding_names=["y"],
            name="retention_probe",
            settings=CompilationSettings(lazy_engine_init=True),
        )
        self.assertIs(module._serialized_engine, self.plan)
        module.setup_engine()
        self.assertIsNone(module._serialized_engine)
        self._check(module)

    def test_state_dict_round_trips_twice(self):
        restored = TorchTensorRTModule()
        restored.load_state_dict(self._module().state_dict())
        self._check(restored)
        # A module restored through load_state_dict can be saved again.
        again = TorchTensorRTModule()
        again.load_state_dict(restored.state_dict())
        self._check(again)

    def test_pickle_carries_one_copy_of_the_plan(self):
        buf = io.BytesIO()
        torch.save(self._module(), buf)
        # The engine pickles its plan base64-encoded (4/3 of its size); the module
        # used to pickle a second, raw copy next to it.
        self.assertLess(buf.tell(), 1.6 * len(self.plan))
        buf.seek(0)
        self._check(torch.load(buf, weights_only=False))

    def test_deepcopy(self):
        module = self._module()
        clone = copy.deepcopy(module)
        self.assertIs(clone.engine, module.engine)
        self._check(clone)


if __name__ == "__main__":
    unittest.main()
