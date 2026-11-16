# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from unittest import mock

import torch
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt import Input
from torch_tensorrt.dynamo import partitioning
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

# The multiply runs in PyTorch, so the second engine reads an intermediate
# float64 value, which truncate_double repairs.
PARTITIONED = {"torch_executed_ops": {"torch.ops.aten.mul.Tensor"}}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, x):
        y = (self.linear(x) + 1).to(torch.float64) * 2
        return y.to(torch.float32).relu()


def no_example_tensors():
    made = AssertionError("an example tensor was made for an engine input")
    return mock.patch.object(Input, "example_tensor", side_effect=made)


def profile(low, high):
    return {"min_shape": (low, 8), "opt_shape": (low, 8), "max_shape": (high, 8)}


def num_engines(module):
    return sum(isinstance(m, TorchTensorRTModule) for m in module.modules())


class TestSubmoduleInputs(TestCase):
    @parameterized.expand(
        [
            ("whole_graph", {"require_full_compilation": True}, 1, None),
            ("dynamic", PARTITIONED, 2, "range"),
            ("profiles", PARTITIONED, 2, "profiles"),
        ]
    )
    def test_compile_makes_no_example_tensors_for_engine_inputs(
        self, _, settings, engines, dynamic
    ):
        model = Model().eval().cuda()
        x = torch.randn(4, 8, device="cuda")
        dynamic_shapes = None
        arg_inputs = [x]
        if dynamic:
            batch = torch.export.Dim("batch", min=2, max=16)
            dynamic_shapes = ({0: batch},)
        if dynamic == "profiles":
            # Made before the patch: a user Input keeps its example tensor.
            arg_inputs = [Input(profiles=[profile(2, 4), profile(4, 16)])]
        exported = torch.export.export(model, (x,), dynamic_shapes=dynamic_shapes)
        with no_example_tensors():
            trt_module = torch_tensorrt.dynamo.compile(
                exported,
                arg_inputs=arg_inputs,
                min_block_size=1,
                truncate_double=True,
                **settings,
            )

        self.assertEqual(num_engines(trt_module), engines)
        torch.testing.assert_close(trt_module(x), model(x), rtol=5e-3, atol=5e-3)

    def test_refit_makes_no_example_tensors_for_engine_inputs(self):
        model = Model().eval().cuda()
        x = torch.randn(4, 8, device="cuda")
        trt_module = torch_tensorrt.dynamo.compile(
            torch.export.export(model, (x,)),
            arg_inputs=[x],
            min_block_size=1,
            truncate_double=True,
            immutable_weights=False,
            **PARTITIONED,
        )
        new_model = Model().eval().cuda()
        with no_example_tensors():
            refitted = torch_tensorrt.dynamo.refit_module_weights(
                trt_module, torch.export.export(new_model, (x,))
            )

        self.assertEqual(num_engines(refitted), 2)
        torch.testing.assert_close(refitted(x), new_model(x), rtol=5e-3, atol=5e-3)

    def test_shape_tensor_keeps_its_value(self):
        batch = torch.export.Dim("batch", min=2, max=16)
        exported = torch.export.export(
            torch.nn.ReLU(), (torch.randn(4, 8),), dynamic_shapes=({0: batch},)
        )
        (x,) = [n for n in exported.graph.nodes if n.op == "placeholder"]
        graph = torch.fx.Graph()
        size = graph.placeholder("size")
        size.meta["val"] = x.meta["val"].shape[0]
        graph.output(size)
        submodule = torch.fx.GraphModule(torch.nn.Module(), graph)

        (spec,) = partitioning.construct_submodule_inputs(
            submodule, example_tensors=False
        )

        self.assertTrue(spec.is_shape_tensor)
        self.assertEqual(spec.torch_tensor.tolist(), [spec.shape["opt_shape"][0]])


if __name__ == "__main__":
    run_tests()
