import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import DispatchTestCase, InputTensorSpec


class TestFlattenConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("flatten_middle_dims", 1, 2),
            # ("flatten_last_3_dims", 1, 3),
            # ("flatten_all", 0, 3),
        ]
    )
    def test_flatten(self, _, start_dim, end_dim):
        class Flatten(nn.Module):
            def __init__(self, start, end):
                super().__init__()
                self.start = start
                self.end = end

            def forward(self, x):
                return torch.flatten(x, self.start, self.end)

        inputs = [torch.randn(1, 2, 3, 1)]
        self.run_test(
            Flatten(start_dim, end_dim),
            inputs,
            # This has changed to aten.view instead
            expected_ops=[],
            test_implicit_batch_dim=(start_dim != 0),
        )

    ## Dynamic shape does not work due to flatten converts to reshape in tracing. And batch or dynamic dimension is converted to fixed integer and loose dynamic
    ## For ex., flatten (1, 512, 1, 1) with start_dim=1, end_dim=-1. After convert to reshape, output size=(1, 512) which is not correct since dim=0 is -1.
    ## This problem may be solved using dynamic shape propogation. And we will know dim=0 is dynamic and we should set -1 in converter.

    # @parameterized.expand(
    #     [
    #         ("flatten_middle_dims", 1, 2),
    #     ]
    # )
    # def test_flatten_with_dynamic_shape(self, _, start_dim, end_dim):
    #     class Flatten(nn.Module):
    #         def __init__(self, start, end):
    #             super().__init__()
    #             self.start = start
    #             self.end = end

    #         def forward(self, x):
    #             return torch.flatten(x, self.start, self.end)

    #     input_specs = [
    #         InputTensorSpec(
    #             shape=(-1, -1, -1, -1, -1),
    #             dtype=torch.float32,
    #             shape_ranges=[((1, 1, 1, 1, 1), (1, 2, 3, 2, 1), (3, 3, 3, 3, 3))],
    #         ),
    #     ]
    #     self.run_test_with_dynamic_shape(
    #         Flatten(start_dim, end_dim),
    #         input_specs,
    #         expected_ops={torch.ops.aten._reshape_alias.default},
    #     )


if __name__ == "__main__":
    run_tests()
