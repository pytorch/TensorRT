import fx2trt_oss.tracer.acc_tracer.acc_ops as acc_ops
import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_fx2trt import AccTestCase
from torch.testing._internal.common_utils import run_tests


class TestInterpolateConverter(AccTestCase):
    @parameterized.expand(
        [
            # 3D
            ("3d_dim_scale", (2, 3, 4), (None), (2), ("nearest"), (None)),
            ("3d_dim_scale_seq", (2, 3, 4), (None), (2,), ("nearest"), (None)),
            ("3d_dim_size", (2, 3, 4), (2), (None), ("nearest"), (None)),
            ("3d_dim_size_seq", (2, 3, 4), (8,), (None), ("nearest"), (None)),
            (
                "3d_dim_scale_linear",
                (2, 3, 4),
                (None),
                (2),
                ("linear"),
                (None),
            ),  # linear for 3D only
            (
                "3d_dim_scale_align",
                (2, 3, 4),
                (None),
                (2),
                ("linear"),
                (True),
            ),  # align_corners for linear,bilinear,trilinear,bicubic only
            # 4D
            ("4d_dim_scale", (2, 3, 4, 5), (None), (2), ("nearest"), (None)),
            ("4d_dim_scale_seq", (2, 3, 4, 5), (None), (2, 2), ("nearest"), (None)),
            ("4d_dim_size", (2, 3, 4, 5), (2), (None), ("nearest"), (None)),
            ("4d_dim_size_seq", (2, 3, 4, 5), (8, 10), (None), ("nearest"), (None)),
            (
                "4d_dim_scale_bilinear",
                (2, 3, 4, 5),
                (None),
                (2),
                ("bilinear"),
                (None),
            ),  # linear for 4D only
            (
                "4d_dim_scale_align",
                (2, 3, 4, 5),
                (None),
                (2),
                ("bilinear"),
                (True),
            ),  # align_corners for linear,bilinear,trilinear,bicubic only
            # 5D
            ("5d_dim_scale", (2, 3, 4, 5, 6), (None), (2), ("nearest"), (None)),
            (
                "5d_dim_scale_seq",
                (2, 3, 4, 5, 6),
                (None),
                (2, 2, 2),
                ("nearest"),
                (None),
            ),
            ("5d_dim_size", (2, 3, 4, 5, 6), (2), (None), ("nearest"), (None)),
            (
                "5d_dim_size_seq",
                (2, 3, 4, 5, 6),
                (8, 10, 12),
                (None),
                ("nearest"),
                (None),
            ),
            (
                "5d_dim_scale_trilinear",
                (2, 3, 4, 5, 6),
                (None),
                (2),
                ("trilinear"),
                (None),
            ),  # trilinear for 5D only
            (
                "5d_dim_scale_align",
                (2, 3, 4, 5, 6),
                (None),
                (2),
                ("trilinear"),
                (True),
            ),  # align_corners for linear,bilinear,trilinear,bicubic only
        ]
    )
    def test_interpolate(self, _, init_size, size, scale_factor, mode, align_corners):
        class Interpolate(nn.Module):
            def forward(self, x):
                return torch.nn.functional.interpolate(
                    x,
                    size=size,
                    scale_factor=scale_factor,
                    mode=mode,
                    align_corners=align_corners,
                )  # only one of size or scale_factor should be defined

        inputs = [torch.randn(*init_size)]
        self.run_test(
            Interpolate(),
            inputs,
            expected_ops={acc_ops.interpolate},
        )


if __name__ == "__main__":
    run_tests()
