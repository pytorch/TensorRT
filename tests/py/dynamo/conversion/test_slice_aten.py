import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestSliceConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("slice_dim_start_stop_step", 0, 0, 7, 2),
            ("slice_dim_start_stop_step_offset", 1, 0, 7, 2),
            ("slice_dim_start_stop_step_exact", 1, 0, 10, 2),
            ("slice_dim_start_stop_step_negatives", -3, -2, -1, 1),
            ("slice_dim_start_stop_step_max_int", 2, 0, 2**63 - 1, 1),
            ("slice_dim_start_stop_step_past_end", 2, 0, 2048, 1),
            ("slice_dim_start_stop_step_none", 2, None, None, 1),
        ]
    )
    def test_slice(self, _, dim, start, stop, step):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                out = torch.ops.aten.slice.Tensor(input, dim, start, stop, step)
                return out

        input = [torch.randn(10, 10, 3, 1)]
        self.run_test(
            TestModule(),
            input,
        )

    def test_slice_empty(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                out = torch.ops.aten.slice.Tensor(input)
                return out

        input = [torch.randn(10, 10, 3, 1)]
        self.run_test(
            TestModule(),
            input,
        )


class TestSliceConverterDynamicShape(DispatchTestCase):
    @parameterized.expand(
        [
            (
                "slice_dynamic_dim_start_stop_step_offset",
                (1, 10, 1),
                (1, 10, 10),
                (1, 10, 10),
                1,
                0,
                7,
                2,
            ),
            (
                "slice_dynamic_dim_start_stop_step",
                (1, 10, 1),
                (1, 10, 10),
                (1, 10, 10),
                1,
                0,
                10,
                2,
            ),
            (
                "slice_dynamic_dim_start_stop_step_negatives",
                (1, 10, 10),
                (10, 10, 10),
                (10, 10, 10),
                -2,
                -2,
                -1,
                1,
            ),
            (
                "slice_dim_start_stop_step_max_int",
                (1, 10, 10),
                (10, 10, 10),
                (10, 10, 10),
                2,
                0,
                2**63 - 1,
                1,
            ),
            (
                "slice_dim_start_stop_step_past_end",
                (1, 10, 10),
                (10, 10, 10),
                (10, 10, 10),
                2,
                0,
                2048,
                1,
            ),
            (
                "slice_dim_start_stop_step_none",
                (1, 10, 10),
                (10, 10, 10),
                (10, 10, 10),
                2,
                None,
                None,
                1,
            ),
            (
                "slice_dynamic_dim_start_stop_step_offset_4D",
                (1, 10, 1, 3),
                (1, 10, 10, 3),
                (1, 10, 10, 3),
                1,
                0,
                7,
                2,
            ),
            (
                "slice_dynamic_dim_start_stop_step_4D",
                (1, 10, 1, 3),
                (1, 10, 10, 3),
                (1, 10, 10, 3),
                1,
                0,
                10,
                2,
            ),
            (
                "slice_dynamic_dim_dyn_start_dyn_stop_step",
                (1, 10, 1),
                (1, 10, 10),
                (1, 10, 10),
                2,
                -2,
                10,
                2,
            ),
            (
                "slice_dynamic_dim_dyn_start_stop_dyn_step",
                (1, 10, 1),
                (1, 10, 10),
                (1, 10, 10),
                2,
                0,
                -2,
                2,
            ),
            (
                "slice_dynamic_dim_dyn_start_stop_None_step",
                (1, 10, 1),
                (1, 10, 10),
                (1, 10, 10),
                2,
                0,
                None,
                2,
            ),
            (
                "slice_dynamic_dim_dyn_start_dyn_stop_dyn_step",
                (1, 10, 1),
                (1, 10, 10),
                (1, 10, 10),
                2,
                -8,
                -2,
                2,
            ),
            (
                "slice_dynamic_dim_dyn_start_dyn_stop_dyn_step_ceil",
                (1, 10, 1),
                (1, 10, 10),
                (1, 10, 10),
                2,
                -9,
                -2,
                2,
            ),
            (
                "slice_dynamic_dim_dyn_start_dyn_stop_dyn_step_diff_dim",
                (1, 10, 1),
                (1, 10, 10),
                (1, 10, 10),
                0,
                -8,
                -2,
                2,
            ),
            (
                "slice_dynamic_dim_dyn_start_dyn_stop_dyn_step_diff_dim_ceil",
                (1, 10, 1),
                (1, 10, 10),
                (1, 10, 10),
                0,
                -9,
                -2,
                2,
            ),
        ]
    )
    def test_slice(self, _, min_shape, opt_shape, max_shape, dim, start, stop, step):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                out = torch.ops.aten.slice.Tensor(input, dim, start, stop, step)
                return out

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=torch.float32,
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(),
            input_specs,
        )

    def test_slice_dynamic_computed_negative_start(self):
        # Regression test for https://github.com/pytorch/TensorRT/issues/4186.
        # Unlike the cases above (a literal negative int start, known at
        # trace time), here `start` is computed from the input's own dynamic
        # shape, so by the time it reaches impl.slice.slice_op it is a
        # TRTTensor whose value (and sign) is only known at runtime. Every
        # op feeding it (sym_size, mul, neg) has a converter, so this stays
        # in a single TRT engine with no PyTorch fallback. Provably negative
        # for every shape in the profile (x.shape[0] * 2, negated), so
        # torch.export doesn't need to guard/specialize on its sign the way
        # it would for something like -(x.shape[0] // 2).
        class TestModule(torch.nn.Module):
            def forward(self, x):
                k = -(x.shape[0] * 2)
                return torch.ops.aten.slice.Tensor(x, 0, k, None, 1)

        input_specs = [
            Input(
                min_shape=(2, 3),
                opt_shape=(6, 3),
                max_shape=(64, 3),
                dtype=torch.float32,
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(),
            input_specs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )


if __name__ == "__main__":
    run_tests()
