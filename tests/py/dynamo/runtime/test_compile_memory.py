# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for how much host and device memory compilation holds on to."""

import os
import platform
import unittest
from unittest import mock

import torch
from torch_tensorrt.dynamo import utils


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


if __name__ == "__main__":
    unittest.main()
