import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.dynamo.tools.common_fx2trt import DispatchTestCase, InputTensorSpec


class TestLinearConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("default", [1, 512], True, torch.ops.aten.linear),
            ("matrix", [5, 512], True, torch.ops.aten.linear),
            ("no_bias", [1, 512], False, torch.ops.aten.linear),
            (
                "multi_dim_matrix",
                [4, 5, 512],
                True,
                torch.ops.aten.linear,
            ),
            (
                "multi_dim_matrix",
                [4, 5, 512],
                False,
                torch.ops.aten.linear,
            ),
        ]
    )
    def test_linear(self, test_name, shape, bias, op):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(512, 256, bias)

            def forward(self, x):
                return self.linear(x)

        inputs = [torch.randn(shape)]
        self.run_test(TestModule(), inputs, expected_ops={op})

        # linear will be decomposed to P531484488 and view(reshape) can not handle reshape pattern
        # like (2, 3, n)->(6, n) in implicit mode which is similar to dynamic shape test below.

    # Input is transposed through view [3,3,512]->[9,512]. Converter does not know dim=0 is dynamic now.

    # def test_linear_with_dynamic_shape(self):
    #     class TestModule(torch.nn.Module):
    #         def __init__(self):
    #             super().__init__()
    #             self.linear = torch.nn.Linear(512, 256)

    #         def forward(self, x):
    #             return self.linear(x)

    #     input_specs = [
    #         InputTensorSpec(
    #             shape=(-1, 3, 512),
    #             dtype=torch.float32,
    #             shape_ranges=[((1, 3, 512), (3, 3, 512), (4, 3, 512))],
    #         ),
    #     ]
    #     self.run_test_with_dynamic_shape(
    #         TestModule(),
    #         input_specs,
    #         expected_ops={torch.ops.aten.addmm.default},
    #     )

    ## Testing with (-1, -1, 512) results into following error:
    ## AssertionError: Currently we only support one dynmaic dim for linear and it can't be the last dim.


if __name__ == "__main__":
    run_tests()
