import torch
import torch.nn as nn
from harness import DispatchTestCase
from parameterized import param, parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input


class TestIndexSelectConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("1d_input", (10,), 0, (1,)),
            ("2d_input_dim_0", (10, 3), 0, (0, 2)),
            ("2d_input_dim_1", (5, 10), 1, (1, 2, 3)),
            ("3d_input_dim_0", (10, 5, 10), 0, (0, 5)),
            ("3d_input_dim_2", (10, 5, 10), 2, (3, 3, 4)),
        ]
    )
    def test_index_select(self, _, source_shape, dim, indices_val):
        class TestIndexSelect(torch.nn.Module):
            def forward(self, source_tensor, indices_tensor):
                return torch.ops.aten.index_select.default(
                    source_tensor, dim, indices_tensor
                )

        input = [
            torch.randn(*source_shape, dtype=torch.float32),
            torch.tensor([*indices_val], dtype=torch.int32),
        ]

        self.run_test(
            TestIndexSelect(),
            input,
        )


if __name__ == "__main__":
    run_tests()
