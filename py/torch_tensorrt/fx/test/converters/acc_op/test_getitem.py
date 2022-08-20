import torch
import torch.nn as nn
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec


class TestGetitemConverter(AccTestCase):
    @parameterized.expand(
        [
            ("slice_batch_dim", slice(None, None, None)),
            ("slice_basic", (slice(None, None, None), slice(0, 3, 2))),
            ("slice_full", (slice(None, None, None), slice(0, 10, 3))),
            ("ellipsis", (slice(None, None, None), ..., slice(0, 3, 2))),
            (
                "slice_all_none",
                (slice(None, None, None), slice(None, None, None)),
            ),
            (
                "slice_start_none",
                (slice(None, None, None), slice(None, 2, 1)),
            ),
            ("slice_end_none", (slice(None, None, None), slice(1, None, 1))),
            (
                "slice_step_none",
                (slice(None, None, None), slice(0, 3, None)),
            ),
            ("slice_neg_idx", (slice(None, None, None), -1)),
            ("slice_neg_slice", (slice(None, None, None), slice(-8, -2, 3))),
            ("multi_dim", (slice(None, None, None), 0, 1)),
            (
                "slice_multi_dim",
                (slice(None, None, None), slice(0, 3, 2), slice(1, -1, 3)),
            ),
            (
                "none",
                (slice(None, None, None), None, slice(1, -1, 3), 1),
            ),
            (
                "slice_zero_slice",
                (slice(None, None, None), slice(None, None, None), slice(0, 0, None)),
            ),
        ]
    )
    def test_getitem(self, _, idx):
        class Getitem(nn.Module):
            def __init__(self, idx):
                super().__init__()
                self.idx = idx

            def forward(self, x):
                x = x + x
                return x[self.idx]

        inputs = [torch.randn(2, 10, 10, 10)]
        self.run_test(Getitem(idx), inputs, expected_ops={acc_ops.getitem})

    @parameterized.expand(
        [
            ("slice_batch_dim", slice(None, None, None)),
            ("ellipsis", (slice(None, None, None), ..., slice(0, -3, 2))),
            (
                "slice_all_none",
                (slice(None, None, None), slice(None, None, None)),
            ),
            (
                "slice_end_none",
                (slice(None, None, None), slice(None, None, None), slice(1, None, 1)),
            ),
            (
                "slice_step_none",
                (slice(None, None, None), slice(None, None, None), slice(0, 3, None)),
            ),
            ("slice_neg_idx", (slice(None, None, None), -1, slice(None, None, None))),
            (
                "slice_neg_slice",
                (slice(None, None, None), slice(None, None, None), slice(-8, -2, 3)),
            ),
            ("multi_dim", (slice(None, None, None), 0, 1)),
            (
                "slice_multi_dim",
                (slice(None, None, None), slice(0, 3, 2), slice(1, -1, 3)),
            ),
            (
                "none",
                (slice(None, None, None), None, slice(1, -1, 3)),
            ),
        ]
    )
    def test_getitem_with_dynamic_shape(self, _, idx):
        class Getitem(nn.Module):
            def __init__(self, idx):
                super().__init__()
                self.idx = idx

            def forward(self, x):
                x = x + x
                return x[self.idx]

        input_specs = [
            InputTensorSpec(
                shape=(-1, 256, 256),
                dtype=torch.float32,
                shape_ranges=[((1, 256, 256), (3, 256, 256), (5, 256, 256))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            Getitem(idx), input_specs, expected_ops={acc_ops.getitem}
        )

    @parameterized.expand(
        [
            ("slice_batch_dim", slice(None, None, None)),
            ("ellipsis", (slice(None, None, None), ..., slice(0, -3, 2))),
            (
                "slice_all_none",
                (slice(None, None, None), slice(None, None, None)),
            ),
            (
                "slice_end_none",
                (slice(None, None, None), slice(None, None, None), slice(1, None, 1)),
            ),
            (
                "slice_step_none",
                (slice(None, None, None), slice(None, None, None), slice(0, 3, None)),
            ),
        ]
    )
    def test_getitem_with_multi_dynamic_shape(self, _, idx):
        class Getitem(nn.Module):
            def __init__(self, idx):
                super().__init__()
                self.idx = idx

            def forward(self, x):
                x = x + x
                return x[self.idx]

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, 256),
                dtype=torch.float32,
                shape_ranges=[((1, 128, 256), (3, 192, 256), (5, 256, 256))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            Getitem(idx), input_specs, expected_ops={acc_ops.getitem}
        )

    # Testing with following parameters results into Error:
    # AssertionError: We don't support slicing tensor on dynamic shape.
    """
        ("ellipsis", (slice(None, None, None), ..., slice(0, -3, 2))),
        (
            "slice_end_none",
            (slice(None, None, None), slice(None, None, None), slice(1, None, 1)),
        ),
        (
            "slice_step_none",
            (slice(None, None, None), slice(None, None, None), slice(0, 3, None)),
        ),
    """

    @parameterized.expand(
        [
            ("slice_batch_dim", slice(None, None, None)),
            (
                "slice_all_none",
                (slice(None, None, None), slice(None, None, None)),
            ),
        ]
    )
    def test_getitem_with_dynamic_shape_four_dimensions(self, _, idx):
        class Getitem(nn.Module):
            def __init__(self, idx):
                super().__init__()
                self.idx = idx

            def forward(self, x):
                x = x + x
                return x[self.idx]

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (3, 3, 3, 3), (5, 5, 5, 5))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            Getitem(idx), input_specs, expected_ops={acc_ops.getitem}
        )


if __name__ == "__main__":
    run_tests()
