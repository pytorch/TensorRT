import torch
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import parameterized
from torch import nn
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec


class TestRepeatInterLeave(AccTestCase):
    @parameterized.expand(
        [
            ("none_dim", (2, 3, 4), 3, None),
            ("dim_0", (2, 3, 4), 3, 0),
            ("dim_1", (2, 3, 4), 3, 1),
            ("dim_2", (2, 3, 4), 3, 2),
        ]
    )
    def test_repeat_interleave(self, _, input_shape, repeat, dim):
        class RepeatInterleave(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.repeat = repeat
                self.dim = dim

            def forward(self, x):
                return torch.repeat_interleave(x, self.repeat, self.dim)

        inputs = [torch.randn(*input_shape)]
        expected_ops = {acc_ops.tile, acc_ops.unsqueeze, acc_ops.reshape}
        if dim is not None:
            expected_ops.update({acc_ops.getitem, acc_ops.size})
        self.run_test(
            RepeatInterleave(dim),
            inputs,
            expected_ops=expected_ops,
            test_implicit_batch_dim=dim is not None and dim != 0,
        )

    @parameterized.expand(
        [
            ("none_dim", (-1, 2, 3), 3, None),
            ("dim_0", (-1, 2, 3), 3, 0),
            ("dim_1", (-1, 2, 3), 3, 1),
            ("dim_2", (-1, 3, 2), 3, 2),
        ]
    )
    def test_repeat_interleave_with_dynamic_shape(self, _, input_shape, repeat, dim):
        class RepeatInterleave(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.repeat = repeat
                self.dim = dim

            def forward(self, x):
                return torch.repeat_interleave(x, self.repeat, self.dim)

        input_specs = [
            InputTensorSpec(
                shape=input_shape,
                dtype=torch.float32,
                shape_ranges=[
                    (
                        tuple(i if i != -1 else 1 for i in input_shape),
                        tuple(i if i != -1 else 2 for i in input_shape),
                        tuple(i if i != -1 else 3 for i in input_shape),
                    )
                ],
            ),
        ]
        self.run_test_with_dynamic_shape(
            RepeatInterleave(dim), input_specs, expected_ops={acc_ops.tile}
        )


if __name__ == "__main__":
    run_tests()
