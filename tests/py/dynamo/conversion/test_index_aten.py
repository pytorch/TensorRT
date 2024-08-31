import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestIndexConstantConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (
                "index_zero_two_dim_indices_input",
                [None, torch.randint(0, 1, (1, 1))],
                torch.randn(2, 2),
            ),
            (
                "index_zero_three_dim_indices_input",
                [None, torch.randint(0, 1, (1, 1)), None],
                torch.randn(2, 2, 2),
            ),
            (
                "index_zero_index_one_three_dim_indices_input",
                [None, torch.randint(0, 1, (1, 1)), torch.randint(0, 1, (1, 1))],
                torch.randn(2, 2, 2),
            ),
            (
                "index_zero_index_one_four_dim_indices_input",
                [None, torch.tensor([0, 0, 1, 1]), torch.tensor([0, 0, 1, 1]), None],
                torch.randn(2, 4, 4, 2),
            ),
            (
                "index_zero_index_one_four_dim_indices_input_SD",
                [
                    None,
                    torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]),
                    torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]),
                    None,
                ],
                torch.randn(2, 1280, 8, 8),
            ),
            (
                "index_zero_index_one_four_dim_indices_input_SD_unsqueeze",
                [
                    None,
                    torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])
                    .unsqueeze(0)
                    .T.long(),
                    torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])
                    .unsqueeze(0)
                    .T.long(),
                    None,
                ],
                torch.randn(2, 1280, 8, 8),
            ),
            (
                "index_zero_index_one_four_dim_indices_input_SD_unsqueeze_broadcast",
                [
                    None,
                    torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]),
                    torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])
                    .unsqueeze(0)
                    .T.long(),
                    None,
                ],
                torch.randn(2, 1280, 8, 8),
            ),
            (
                "index_zero_index_one_four_dim_indices_input_non_continuous",
                [None, torch.tensor([0, 0, 1, 1]), None, torch.tensor([0, 0, 1, 1])],
                torch.randn(2, 4, 4, 2),
            ),
        ]
    )
    def test_index_constant(self, _, index, input):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.index.Tensor(input, index)

        inputs = [input]
        self.run_test(TestModule(), inputs)


# The below tests cannot be included in the parameterized
# [None, index0] cannot be passed as torch.Tensor to DispatchTestCase.run_test()
# tensorrt.Input requires the input to be torch Tensor
class TestIndexConverter(DispatchTestCase):
    def test_index_zero_two_dim_ITensor(self):
        class TestModule(nn.Module):
            def forward(self, x, index0):
                indices = [None, index0]
                out = torch.ops.aten.index.Tensor(x, indices)
                return out

        input = torch.randn(2, 2)
        index0 = torch.randint(0, 1, (1, 1))
        index0 = index0.to(torch.int32)
        self.run_test(
            TestModule(),
            [input, index0],
        )

    def test_index_zero_index_three_dim_ITensor(self):
        class TestModule(nn.Module):
            def forward(self, x, index0):
                indices = [None, index0, None]
                out = torch.ops.aten.index.Tensor(x, indices)
                return out

        input = torch.randn(2, 2, 2)
        index0 = torch.randint(0, 1, (1, 1))
        index0 = index0.to(torch.int32)
        self.run_test(TestModule(), [input, index0])


class TestIndexDynamicConstantConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (
                "index_zero_two_dim_indices_input_min_opt_max",
                [None, torch.randint(0, 1, (1, 1))],
                (2, 1),
                (2, 2),
                (2, 2),
            ),
            (
                "index_zero_three_dim_indices_input_min_opt_max",
                [None, torch.randint(0, 1, (1, 1)), None],
                (2, 1, 2),
                (2, 2, 2),
                (2, 2, 2),
            ),
            (
                "index_zero_index_one_three_dim_indices_input_min_opt_max",
                [None, torch.randint(0, 1, (1, 1)), torch.randint(0, 1, (1, 1))],
                (2, 1, 2),
                (2, 2, 2),
                (2, 2, 2),
            ),
            (
                "index_zero_index_one_four_dim_indices_input_min_opt_max",
                [None, torch.tensor([0, 0, 1, 1]), torch.tensor([0, 0, 1, 1]), None],
                (2, 1, 4, 2),
                (2, 4, 4, 2),
                (2, 4, 4, 2),
            ),
        ]
    )
    def test_index_constant_dynamic(
        self, _, index, input_min_shape, input_opt_shape, input_max_shape
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.index.Tensor(input, index)

        input_specs = [
            Input(
                min_shape=input_min_shape,
                opt_shape=input_opt_shape,
                max_shape=input_max_shape,
                dtype=torch.float32,
            ),
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)


if __name__ == "__main__":
    run_tests()
