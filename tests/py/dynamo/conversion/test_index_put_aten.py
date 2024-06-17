import torch
import torch.nn as nn
from parameterized import param, parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestIndexPutConverter(DispatchTestCase):
    @parameterized.expand(
        [
            param(
                test_name="1d_indices_single",
                source_tensor=torch.zeros([5], dtype=torch.int32),
                indices_tensor=(torch.tensor([0], dtype=torch.int32),),
                value_tensor=torch.tensor([1], dtype=torch.int32),
            ),
            param(
                test_name="1d_indices_multiple",
                source_tensor=torch.zeros([5], dtype=torch.int32),
                indices_tensor=(torch.tensor([0, 3], dtype=torch.int32),),
                value_tensor=torch.tensor([1, 3], dtype=torch.int32),
            ),
            # param(
            #     test_name="2d_indices",
            #     source_tensor=torch.zeros([5,5], dtype=torch.int32),
            #     indices_tensor=(torch.tensor([0,2], dtype=torch.int32),torch.tensor([2,0], dtype=torch.int32),),
            #     value_tensor=torch.tensor([1,3], dtype=torch.int32),
            # ),
        ]
    )
    def test_index_put(
        self, test_name, source_tensor, indices_tensor, value_tensor, accumulate=True
    ):
        class TestIndexPut(torch.nn.Module):
            def forward(self, source_tensor, value_tensor):
                return torch.ops.aten.index_put_.default(
                    source_tensor, indices_tensor, value_tensor
                )

        self.run_test(
            TestIndexPut(),
            inputs=[source_tensor, value_tensor],
        )


if __name__ == "__main__":
    run_tests()
