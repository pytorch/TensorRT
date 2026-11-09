# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import unittest.mock as mock

import torch
import torch.nn as nn
import torch_tensorrt as torchtrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo import partitioning
from torch_tensorrt.dynamo.conversion import CompilationSettings
from torch_tensorrt.dynamo.lowering import get_decompositions
from torch_tensorrt.dynamo.lowering.passes import post_lowering, pre_export_lowering
from torch_tensorrt.dynamo.partitioning._resource_partitioner import resource_partition

_MiB = 1024 * 1024


class _Chain(nn.Module):
    """Eight 512x512 fp32 linears: 1 MiB of weights each."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(512, 512, bias=False) for _ in range(8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


def _capability_partitioned() -> torch.fx.GraphModule:
    settings = CompilationSettings(min_block_size=1, enable_resource_partitioning=True)
    ep = torch.export.export(_Chain().eval().cuda(), (torch.randn(4, 512).cuda(),))
    ep = pre_export_lowering(ep, settings).run_decompositions(get_decompositions(False))
    gm = post_lowering(ep.module(), settings)
    partitioned, _ = partitioning.fast_partition(gm, min_block_size=1, skip_fusion=True)
    return partitioned


def _engines(rss: int, **kwargs: int) -> int:
    memory_info = mock.MagicMock()
    memory_info.rss = rss
    with mock.patch("psutil.Process") as process:
        process.return_value.memory_info.return_value = memory_info
        gm = resource_partition(_capability_partitioned(), **kwargs)
    return sum("_run_on_acc" in name for name, _ in gm.named_children())


class TestMaxPartitionSize(TestCase):
    def test_default_and_old_pickle(self) -> None:
        self.assertIsNone(CompilationSettings().max_partition_size)
        state = CompilationSettings().__dict__.copy()
        state.pop("max_partition_size")
        restored = CompilationSettings.__new__(CompilationSettings)
        restored.__setstate__(state)
        self.assertIsNone(restored.max_partition_size)

    def test_budget_split_depends_on_rss(self) -> None:
        budget = 64 * _MiB
        self.assertNotEqual(
            _engines(rss=budget - 16 * _MiB, cpu_memory_budget=budget),
            _engines(rss=budget - 32 * _MiB, cpu_memory_budget=budget),
        )

    def test_size_split_ignores_rss(self) -> None:
        splits = {
            _engines(rss=rss, cpu_memory_budget=None, max_partition_size=2 * _MiB)
            for rss in (16 * _MiB, 1024 * _MiB, 64 * 1024 * _MiB)
        }
        # Eight 1 MiB layers at a 2 MiB size: four engines of 2 MiB.
        self.assertEqual(splits, {4})

    def test_size_takes_precedence_over_budget(self) -> None:
        self.assertEqual(_engines(rss=0, cpu_memory_budget=1024 * _MiB), 1)
        self.assertGreater(
            _engines(rss=0, cpu_memory_budget=1024 * _MiB, max_partition_size=2 * _MiB),
            1,
        )

    def test_size_above_model_size_keeps_one_engine(self) -> None:
        self.assertEqual(
            _engines(rss=0, cpu_memory_budget=None, max_partition_size=64 * _MiB), 1
        )

    def test_non_positive_size_is_rejected(self) -> None:
        for size in (0, -1):
            with self.assertRaisesRegex(ValueError, "positive"):
                CompilationSettings(max_partition_size=size)

    def test_size_turns_resource_partitioning_on(self) -> None:
        settings = CompilationSettings(max_partition_size=_MiB)
        self.assertTrue(settings.enable_resource_partitioning)

    def test_size_and_budget_together_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not both"):
            CompilationSettings(
                max_partition_size=_MiB, cpu_memory_budget=8 * 1024 * _MiB
            )

    def test_size_too_small_names_the_setting_and_its_unit(self) -> None:
        with self.assertRaisesRegex(ValueError, r"max_partition_size is 16384 bytes"):
            _engines(rss=0, cpu_memory_budget=None, max_partition_size=_MiB // 64)

    def test_compile_with_max_partition_size(self) -> None:
        torch.manual_seed(0)
        model = _Chain().eval().cuda()
        x = torch.randn(4, 512).cuda()
        trt = torchtrt.dynamo.compile(
            torch.export.export(model, (x,)),
            inputs=[x],
            min_block_size=1,
            max_partition_size=2 * _MiB,
        )
        engines = [n for n, _ in trt.named_children() if "_run_on_acc" in n]
        self.assertEqual(len(engines), 4)
        with torch.no_grad():
            expected = model(x)
            # Eight layers shrink the activations to about 1e-3, so the tolerance follows their scale.
            torch.testing.assert_close(
                trt(x), expected, rtol=1e-2, atol=1e-2 * expected.abs().max().item()
            )
        torch._dynamo.reset()


if __name__ == "__main__":
    run_tests()
