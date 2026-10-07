# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import operator
import unittest
from unittest.mock import patch

import torch
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_ATEN_CONVERTERS,
    DYNAMO_CONVERTERS,
    ConverterSupport,
    node_has_uint8_tensors,
)


def _exported_node(model, inputs, target):
    # Decomposed as compile() does, so .float() appears as the _to_copy converters see it.
    graph = torch.export.export(model, inputs).run_decompositions().graph
    (node,) = [node for node in graph.nodes if node.target == target]
    return node


class TestUint8ConverterSupport(TestCase):
    """TensorRT accepts a uint8 tensor only into an identity, a cast or a shape read,
    so only converters that declare supports_uint8 may take one."""

    frame = torch.randint(0, 256, (4, 6, 3), dtype=torch.uint8)

    @parameterized.expand(
        [
            ("permute", lambda x: x.permute(2, 0, 1), torch.ops.aten.permute.default),
            ("compare", lambda x: x > 100, torch.ops.aten.gt.Scalar),
            ("uint8_result", lambda x: x + 2, torch.ops.aten.add.Tensor),
        ]
    )
    def test_rejected(self, _, function, target):
        class Model(torch.nn.Module):
            def forward(self, x):
                return function(x)

        node = _exported_node(Model(), (self.frame,), target)
        self.assertTrue(node_has_uint8_tensors(node))
        self.assertNotIn(node, DYNAMO_CONVERTERS)

    @parameterized.expand(
        [
            ("cast", lambda x: x.float(), torch.ops.aten._to_copy.default),
            ("floating_result", lambda x: x / 255.0, torch.ops.aten.div.Tensor),
        ]
    )
    def test_accepted(self, _, function, target):
        class Model(torch.nn.Module):
            def forward(self, x):
                return function(x)

        node = _exported_node(Model(), (self.frame,), target)
        self.assertTrue(node_has_uint8_tensors(node))
        self.assertIn(node, DYNAMO_CONVERTERS)

    def test_flag_admits_a_converter(self):
        class Permute(torch.nn.Module):
            def forward(self, x):
                return x.permute(2, 0, 1)

        node = _exported_node(Permute(), (self.frame,), torch.ops.aten.permute.default)
        support = ConverterSupport(
            converter_implementation=lambda *args: None, supports_uint8=True
        )
        with patch.dict(
            DYNAMO_ATEN_CONVERTERS, {torch.ops.aten.permute.default: [support]}
        ):
            self.assertIn(node, DYNAMO_CONVERTERS)

    def test_constants_and_non_aten_operators_are_not_counted(self):
        # FP4 weights arrive packed in uint8 constants, and non-ATen operators
        # handle their own types.
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.empty(4)
        packed = graph.get_attr("packed")
        packed.meta["val"] = torch.empty(4, dtype=torch.uint8)
        weighted = graph.call_function(torch.ops.aten.mul.Tensor, (x, packed))
        weighted.meta["val"] = torch.empty(4)
        frame = graph.placeholder("frame")
        frame.meta["val"] = torch.empty(4, dtype=torch.uint8)
        python_add = graph.call_function(operator.add, (frame, frame))

        self.assertFalse(node_has_uint8_tensors(weighted))
        self.assertFalse(node_has_uint8_tensors(python_add))


class CameraFrames(torch.nn.Module):
    """Raw camera frames as a robot policy takes them: uint8 HWC, rearranged and
    thresholded before the cast to float."""

    def forward(self, front, wrist):
        frames = torch.stack([front, wrist]).permute(0, 3, 1, 2)
        saturated = (front > 200).sum()
        return frames.float() / 255 + saturated.float()


class CastFirst(torch.nn.Module):
    """A cast out of a uint8 input builds into the engine, and so does arithmetic
    that promotes it to a floating type."""

    def forward(self, frame):
        return (frame.float() / 255 + frame / 255.0).permute(2, 0, 1)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestUint8Compilation(TestCase):
    @parameterized.expand([("fast", True), ("global", False)])
    def test_uint8_ops_run_in_pytorch(self, _, use_fast_partitioner):
        model = CameraFrames().eval().cuda()
        front = torch.randint(0, 256, (48, 64, 3), dtype=torch.uint8, device="cuda")
        wrist = torch.randint(0, 256, (48, 64, 3), dtype=torch.uint8, device="cuda")

        compiled = torch_tensorrt.compile(
            model,
            ir="dynamo",
            inputs=[front, wrist],
            min_block_size=1,
            use_fast_partitioner=use_fast_partitioner,
        )

        accelerated = [
            name for name, _ in compiled.named_children() if "_run_on_acc" in name
        ]
        self.assertNotEqual(accelerated, [])
        torch.testing.assert_close(compiled(front, wrist), model(front, wrist))

    @parameterized.expand([("fast", True), ("global", False)])
    def test_cast_out_of_uint8_stays_in_the_engine(self, _, use_fast_partitioner):
        model = CastFirst().eval().cuda()
        frame = torch.randint(0, 256, (48, 64, 3), dtype=torch.uint8, device="cuda")

        compiled = torch_tensorrt.compile(
            model,
            ir="dynamo",
            inputs=[frame],
            min_block_size=1,
            use_fast_partitioner=use_fast_partitioner,
        )

        fallback = [
            name for name, _ in compiled.named_children() if "_run_on_gpu" in name
        ]
        self.assertEqual(fallback, [])
        torch.testing.assert_close(compiled(frame), model(frame))


if __name__ == "__main__":
    run_tests()
