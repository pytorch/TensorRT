# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo import partitioning
from torch_tensorrt.dynamo.conversion._TRTInterpreter import (
    UnsupportedOperatorException,
)


class TestGlobalPartitioning(TestCase):
    @parameterized.expand(
        [
            ({}, 1),
            ({"torch.ops.aten.relu.default"}, 3),
        ]
    )
    def test_end2end_global_partition(self, torch_executed_ops, trt_mod_cnt):
        class SimpleCNN(torch.nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 12, 3, padding=1)
                self.bn = torch.nn.BatchNorm2d(12)
                self.conv2 = torch.nn.Conv2d(12, 12, 3, padding=1)
                self.fc1 = torch.nn.Linear(12 * 56 * 56, 10)

            def forward(self, x, b=5):
                x = self.conv1(x)
                x = F.relu(x)
                x = self.bn(x)
                x = F.max_pool2d(x, (2, 2))
                x = self.conv2(x)
                x = F.relu(x)
                x = F.max_pool2d(x, (2, 2))
                x = torch.flatten(x, 1)
                x = x + b
                return self.fc1(x)

        mod = SimpleCNN().to("cuda")
        mod.eval()
        with torch.no_grad():
            inputs = torch.rand((1, 3, 224, 224)).to("cuda")
            try:
                trt_mod = torch_tensorrt.compile(
                    mod,
                    ir="dynamo",
                    inputs=[inputs],
                    min_block_size=1,
                    torch_executed_ops=torch_executed_ops,
                    use_fast_partitioner=False,
                )
                cnt = 0
                for name, _ in trt_mod.named_children():
                    if "_run_on_acc" in name:
                        cnt += 1
                self.assertEqual(cnt, trt_mod_cnt)
            except Exception as e:
                pytest.fail(f"unexpected exception raised: {e}")

    def test_partition_fully_supported_one_op(self):
        class FullySupportedOneOp(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def forward(self, x, y):
                return torch.ops.aten.add.Tensor(x, y)

        fx_graph = torch.fx.symbolic_trace(FullySupportedOneOp())
        partitioned_graph, _ = partitioning.global_partition(deepcopy(fx_graph))
        self.assertEqual(
            len(list(partitioned_graph.named_children())),
            0,
            "Single operators should not be segmented",
        )

    def test_partition_fully_supported_one_op_require_full_compilation(self):
        class FullySupportedOneOp(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def forward(self, x, y):
                return torch.ops.aten.add.Tensor(x, y)

        fx_graph = torch.fx.symbolic_trace(FullySupportedOneOp())
        partitioned_graph, _ = partitioning.global_partition(
            deepcopy(fx_graph), require_full_compilation=True
        )
        self.assertEqual(
            len(list(partitioned_graph.named_children())),
            1,
            "Single operators can be segmented if full compilation is required",
        )

    def test_partition_fully_supported_multi_op(self):
        class FullySupportedMultiOp(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def forward(self, x, y):
                sum_ = torch.ops.aten.sub.Tensor(x, y)
                concat_ = torch.ops.aten.cat.default(x, sum_)
                relu_ = torch.ops.aten.relu.default(concat_)
                pow_ = torch.ops.aten.pow.Tensor_Scalar(relu_, 2)
                return pow_

        fx_graph = torch.fx.symbolic_trace(FullySupportedMultiOp())
        partitioned_graph, _ = partitioning.global_partition(
            deepcopy(fx_graph), min_block_size=2
        )
        self.assertEqual(
            len(list(partitioned_graph.named_children())),
            1,
            "All operators are supported, there should be one segment",
        )

    @parameterized.expand(
        [
            (["torch.nn.modules.conv.Conv2d"], 2),
            ([], 1),
            (["torch.nn.modules.container.Sequential"], 0),
        ]
    )
    def test_end2end_global_partition_torch_executed_modules(
        self, torch_executed_modules, trt_mod_cnt
    ):
        mod = (
            torch.nn.Sequential(
                torch.nn.Conv2d(3, 8, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(8),
                torch.nn.Conv2d(8, 8, 3, padding=1),
                torch.nn.ReLU(),
            )
            .eval()
            .to("cuda")
        )
        with torch.no_grad():
            inputs = torch.rand((1, 3, 4, 4)).to("cuda")
            trt_mod = torch_tensorrt.compile(
                mod,
                ir="dynamo",
                inputs=[inputs],
                min_block_size=1,
                torch_executed_modules=torch_executed_modules,
                use_fast_partitioner=False,
            )
            cnt = 0
            for name, _ in trt_mod.named_children():
                if "_run_on_acc" in name:
                    cnt += 1
            self.assertEqual(cnt, trt_mod_cnt)

    @parameterized.expand(
        [
            (
                "torch_executed_modules_global",
                {"torch_executed_modules": ["torch.nn.modules.container.Sequential"]},
                False,
            ),
            (
                "torch_executed_modules_fast",
                {"torch_executed_modules": ["torch.nn.modules.container.Sequential"]},
                True,
            ),
            (
                "torch_executed_ops_global",
                {"torch_executed_ops": {"torch.ops.aten.relu.default"}},
                False,
            ),
            (
                "torch_executed_ops_fast",
                {"torch_executed_ops": {"torch.ops.aten.relu.default"}},
                True,
            ),
        ]
    )
    def test_require_full_compilation_with_no_supported_ops(
        self, _, exclusion_kwargs, use_fast_partitioner
    ):
        mod = torch.nn.Sequential(torch.nn.ReLU()).eval().to("cuda")
        inputs = torch.rand((1, 3, 4, 4)).to("cuda")
        with self.assertRaisesRegex(AssertionError, "require_full_compilation"):
            torch_tensorrt.compile(
                mod,
                ir="dynamo",
                inputs=[inputs],
                min_block_size=1,
                require_full_compilation=True,
                use_fast_partitioner=use_fast_partitioner,
                **exclusion_kwargs,
            )

    def test_convert_to_trt_engine_rejects_torch_executed_modules(self):
        mod = (
            torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU())
            .eval()
            .to("cuda")
        )
        inputs = torch.rand((1, 3, 4, 4)).to("cuda")
        exp_program = torch.export.export(mod, (inputs,))
        with self.assertRaises(UnsupportedOperatorException) as ctx:
            torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
                exp_program,
                arg_inputs=[inputs],
                min_block_size=1,
                torch_executed_modules=["torch.nn.modules.conv.Conv2d"],
            )
        # Convolution is normally convertible, so its rejection comes from the exclusion
        self.assertIn("convolution", str(ctx.exception.__cause__))


if __name__ == "__main__":
    run_tests()
