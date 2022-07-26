import torch
import torch.nn as nn
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec


class TestTopKConverter(AccTestCase):
    @parameterized.expand(
        [
            ("top1", 1, -1),
            ("top2", 2, -1),
            ("none_dim", 1, None),
            ("smallest", 1, -1, False),
            ("top1_dim0", 1, 0, False),
        ]
    )
    def test_topk(self, _, k, dim, largest=True):
        class TopK(nn.Module):
            def __init__(self, k, dim):
                super().__init__()
                self.k = k
                self.dim = dim
                self.largest = largest

            def forward(self, x):
                if self.dim is not None:
                    out = torch.topk(
                        x, k=self.k, dim=self.dim, largest=self.largest, sorted=False
                    )
                else:
                    out = torch.topk(x, k=self.k, largest=self.largest, sorted=False)
                return out[0], out[1]

        inputs = [torch.randn(1, 2, 3, 4)]
        self.run_test(
            TopK(k, dim),
            inputs,
            expected_ops={acc_ops.topk},
            test_implicit_batch_dim=(dim != 0),
        )

    @parameterized.expand(
        [
            ("top1", 1, -1),
            ("top2", 2, -1),
            ("none_dim", 1, None),
            ("smallest", 1, -1, False),
            ("top1_dim0", 1, 0, False),
        ]
    )
    def test_topk_with_dynamic_shape_four_dimensions(self, _, k, dim, largest=True):
        class TopK(nn.Module):
            def __init__(self, k, dim):
                super().__init__()
                self.k = k
                self.dim = dim
                self.largest = largest

            def forward(self, x):
                if self.dim is not None:
                    out = torch.topk(
                        x, k=self.k, dim=self.dim, largest=self.largest, sorted=False
                    )
                else:
                    out = torch.topk(x, k=self.k, largest=self.largest, sorted=False)
                return out[0], out[1]

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (3, 3, 3, 3), (3, 3, 3, 3))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            TopK(k, dim), input_specs, expected_ops={acc_ops.topk}
        )


if __name__ == "__main__":
    run_tests()
