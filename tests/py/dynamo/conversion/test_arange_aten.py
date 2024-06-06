import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestArangeConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (0, 5, 1),
            (1, 5, 2),
            (3, 5, 3),
            (5, 0, -1),
            (5, 1, -2),
            (5, 3, -3),
            (5, -2, -1),
            (-5, -2, 2),
            (-5, -3, 1),
            (-2, -5, -1),
        ]
    )
    def test_arange(self, start, end, step):
        class Arange(nn.Module):
            def forward(self, x):
                return torch.ops.aten.arange.start_step(start, end, step)

        inputs = [torch.randn(1, 1)]
        self.run_test(
            Arange(),
            inputs,
            use_dynamo_tracer=True,
        )

    @parameterized.expand(
        [
            (0, (5, 7, 10), (1, 1, 2)),
            # (0, (3, 9, 11), (1, 1, 3)),
        ]
    )
    def test_arange_dynamic(self, start_t, end_t, stride_t):
        class Arange(nn.Module):
            def forward(self, end, stride):
                # Pick the opt values for inference
                return torch.ops.aten.arange.start_step(start_t, end, stride)

        # breakpoint()
        # pyt_input = 7
        inputs = [
            # # start input
            # torch_tensorrt.Input(
            #     min_shape=(start_t[0],),
            #     opt_shape=(start_t[1],),
            #     max_shape=(start_t[2],),
            #     dtype=torch.int64,
            #     torch_tensor=torch.tensor(start_t[1], dtype=torch.int64).cuda(),
            #     is_shape_tensor=True,
            # ),
            # end input
            torch_tensorrt.Input(
                min_shape=(end_t[0],),
                opt_shape=(end_t[1],),
                max_shape=(end_t[2],),
                dtype=torch.int64,
                torch_tensor=torch.tensor(end_t[1], dtype=torch.int64).cuda(),
                is_shape_tensor=True,
            ),
            # stride input
            torch_tensorrt.Input(
                min_shape=(stride_t[0],),
                opt_shape=(stride_t[1],),
                max_shape=(stride_t[2],),
                dtype=torch.int64,
                torch_tensor=torch.tensor(stride_t[1], dtype=torch.int64).cuda(),
                is_shape_tensor=True,
            ),
        ]
        self.run_test_with_dynamic_shape(
            Arange(),
            inputs,
            use_example_tensors=False,
            pyt_inputs=[end_t[1], stride_t[1]],
        )


if __name__ == "__main__":
    run_tests()
