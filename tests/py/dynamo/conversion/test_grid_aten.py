import pytest
import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input
from parameterized import parameterized
from .harness import DispatchTestCase

class TestGridConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("input_grid_interpolation_nearest_sample_fill", [5,5], [5,2], 0, 0),
            ("input_grid_interpolation_nearest_sample_clamp", [5,5], [5,2], 0, 1),
            ("input_grid_interpolation_nearest_sample_reflect", [5,5], [5,2], 0, 2),
            ("input_grid_interpolation_linear_sample_fill", [5,5], [5,2], 1, 0),
            ("input_grid_interpolation_linear_sample_clamp", [5,5], [5,2], 1, 1),
            ("input_grid_interpolation_linear_sample_reflect", [5,5], [5,2], 1, 2),
            ("input_grid_interpolation_cubic_sample_fill", [5,5], [5,2], 2, 0),
            ("input_grid_interpolation_cubic_sample_clamp", [5,5], [5,2], 2, 1),
            ("input_grid_interpolation_cubic_sample_reflect", [5,5], [5,2], 2, 2),
        ]
    )
    def test_grid(self,_, input_shape, dim_shape, interpolation, sample):
        class TestModule(nn.Module):
            def forward(self, x):
                input = torch.randn(10).reshape(input_shape)
                grid = torch.randint(-1, 1, dim_shape)
                return nn.functional.grid(input, grid, interpolation, sample)

        inputs = [torch.randn(1, 10)]
        self.run_test(TestModule(), inputs, expected_ops={torch.ops.aten.grid_sampler.out})




    

    