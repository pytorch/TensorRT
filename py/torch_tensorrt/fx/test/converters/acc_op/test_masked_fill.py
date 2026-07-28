import torch
import torch.nn as nn
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase


class TestMaskedFill(AccTestCase):
    @parameterized.expand(
        [
            ("same_dims", (2, 3), 5),
            ("same_dims_tensor", (2, 3), torch.tensor(5)),
            ("not_same_dims", (2, 1), 5),
            ("not_same_dims_tensor", (2, 1), torch.tensor(5)),
        ]
    )
    def test_masked_fill(self, _, input_shape, value):
        class MaskedFill(nn.Module):
            def __init__(self, input_shape):
                super().__init__()
                self.mask = torch.zeros(input_shape)
                self.mask[0, 0] = 1
                self.mask = self.mask.to(torch.bool)
                self.value = value

            def forward(self, x):
                return x.masked_fill(self.mask, self.value)

        inputs = [torch.ones(*input_shape)]
        self.run_test(
            MaskedFill(input_shape),
            inputs,
            expected_ops={acc_ops.masked_fill},
            test_implicit_batch_dim=False,
        )

    # Testing with (-1, -1, -1, -1) results into following error:
    # RuntimeError: Trying to create tensor with negative dimension -1: [-1, -1, -1, -1]

    @parameterized.expand(
        [
            ("same_dims", (2, 3), (2, 3), 5),
            ("expand_first_dims", (2, 3), (1, 3), 5),
            ("expand_second_dims", (2, 3), (2, 1), 5),
            ("expand_third_dims", (2, 3, 4), (2, 3, 1), 5),
        ]
    )
    def test_masked_fill_expand(self, _, input_shape, mask_shape, value):
        class MaskedFill(nn.Module):
            def __init__(self, input_shape):
                super().__init__()
                self.value = value

            def forward(self, x, mask_input):
                return x.masked_fill(mask_input, self.value)

        mask_input = torch.zeros(*mask_shape)
        index = (0) * len(mask_shape)
        mask_input[index] = 1
        mask_input = mask_input.to(torch.bool)
        inputs = [torch.ones(*input_shape), mask_input]
        self.run_test(
            MaskedFill(input_shape),
            inputs,
            expected_ops={acc_ops.masked_fill},
            test_implicit_batch_dim=False,
        )


if __name__ == "__main__":
    run_tests()
