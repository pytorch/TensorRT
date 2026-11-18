# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import tensorrt as trt
import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion import TRTInterpreter

from .harness import DispatchTestCase, skip_if_trt_rtx_turing


class TestLinearConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (10, 10),
            (10, 100),
            (100, 10),
            (100, 100),
        ]
    )
    def test_linear_converter(self, in_features, out_features):
        skip_if_trt_rtx_turing(self, "nn.Linear (an FP32 GEMM)")

        class LinearModel(nn.Module):
            def __init__(self, in_features, out_features):
                super(LinearModel, self).__init__()
                self.linear = nn.Linear(in_features, out_features)

            def forward(self, x):
                return self.linear(x)

        model = LinearModel(in_features, out_features).eval().cuda()
        inputs = [torch.randn(int(torch.randint(1, 20, (1,))), in_features).cuda()]
        self.run_test(model, inputs, use_dynamo_tracer=True, enable_passes=True)

    def test_linear_with_dynamic_shape(self):
        skip_if_trt_rtx_turing(self, "aten.linear (an FP32 GEMM)")

        class LinearModel(torch.nn.Module):
            def forward(self, x, weight, bias):
                return torch.ops.aten.linear.default(x, weight, bias)

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=(1, 10),
                opt_shape=(10, 10),
                max_shape=(100, 10),
            ),
            Input(dtype=torch.float32, shape=(20, 10)),
            Input(dtype=torch.float32, shape=(20,)),
        ]

        self.run_test_with_dynamic_shape(
            LinearModel(), input_specs, use_dynamo_tracer=True, enable_passes=True
        )

    def test_linear_with_rank_3_input_and_bias(self):
        skip_if_trt_rtx_turing(self, "aten.linear (an FP32 GEMM)")

        class LinearModel(torch.nn.Module):
            def forward(self, x, weight, bias):
                return torch.ops.aten.linear.default(x, weight, bias)

        inputs = [
            torch.randn(2, 3, 4).cuda(),
            torch.randn(5, 4).cuda(),
            torch.randn(5).cuda(),
        ]

        self.run_test(LinearModel(), inputs, use_dynamo_tracer=True, enable_passes=True)

    def test_deduplicate_weights(self):
        class LinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.shared = nn.Linear(4, 6)
                self.single = nn.Linear(4, 5)

            def forward(self, a, b, c):
                return self.shared(a) + self.shared(b) * self.shared(c), self.single(a)

        inputs = [torch.randn(2, 3, 4).cuda() for _ in range(3)]
        graph = self.generate_graph(
            LinearModel().eval().cuda(),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

        def network(deduplicate_weights):
            interpreter = TRTInterpreter(
                graph,
                [Input.from_tensor(i) for i in inputs],
                compilation_settings=CompilationSettings(
                    deduplicate_weights=deduplicate_weights
                ),
            )
            interpreter.run()
            net = interpreter.ctx.net
            layers = [net.get_layer(i) for i in range(net.num_layers)]
            constants = [
                tuple(layer.get_output(0).shape)
                for layer in layers
                if layer.type == trt.LayerType.CONSTANT
            ]
            weights = [
                (layer.get_input(1).name, tuple(layer.get_input(1).shape))
                for layer in layers
                if layer.type == trt.LayerType.MATRIX_MULTIPLY
            ]
            return constants, weights

        constants, weights = network(False)
        self.assertEqual(constants.count((6, 4)), 3)
        self.assertEqual(sorted(s for _, s in weights), [(1, 5, 4)] + [(1, 6, 4)] * 3)

        constants, weights = network(True)
        self.assertEqual(constants.count((6, 4)), 1)
        self.assertEqual(sorted(s for _, s in weights), [(1, 4, 6)] * 3 + [(1, 5, 4)])
        self.assertEqual(len({name for name, s in weights if s == (1, 4, 6)}), 1)


if __name__ == "__main__":
    run_tests()
