# type: ignore
import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestConstantPadConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((1, 2), (1, 1), 0),
            ((2, 1), (2, 1), 1),
            ((3, 4, 2), (1, 2), 2),
            ((3, 4, 2), (1, 2, 3, 1, 2, 3), 0),
            ((3, 3, 4, 2), (1, 2, 3, 4), 0),
            ((3, 3, 4, 2), (1, 2, 3, 4), 2),
            ((3, 3, 4, 2, 1), (1, 2, 3, 4, 5, 1, 2, 3, 4, 5), 0),
            ((3, 3, 4, 2, 1, 2), (1, 2, 3, 4, 1, 2, 3, 4), 4),
        ]
    )
    def test_constant_pad(self, shape, pad, value):
        class TestModule(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.constant_pad_nd.default(input, pad, value)

        input = [torch.randn(shape)]
        self.run_test(
            TestModule(),
            input,
        )


class TestReflectionPadConverter(DispatchTestCase):
    @parameterized.expand(
        [
            # Per pytorch doc, the input should be 2D or 3D
            ((3, 3), (1, 1)),
            ((3, 3), (2, 2)),
            ((2, 2, 2), (1, 1)),
            ((2, 2, 4), (2, 3)),
        ]
    )
    def test_reflection_pad1d(self, shape, padding):
        class TestModule(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.reflection_pad1d.default(input, padding)

        input = [torch.randn(shape)]
        self.run_test(
            TestModule(),
            input,
        )

    @parameterized.expand(
        [
            (
                "3d",
                (1, 1, 1),
                (2, 2, 2),
                (3, 3, 3),
                torch.float,
                (1, 1),
            ),
        ]
    )
    def test_dynamic_shape_reflection_pad1d(
        self, _, min_shape, opt_shape, max_shape, type, padding
    ):
        class reflection_pad1d(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.reflection_pad1d.default(input, padding)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]

        self.run_test_with_dynamic_shape(reflection_pad1d(), input_specs)

    @parameterized.expand(
        [
            # Per pytorch doc, the input should be 3D or 4D
            ((2, 2, 2), (1, 1, 1, 1)),
            ((1, 2, 4), (2, 2, 1, 1)),
            ((2, 2, 3, 3), (1, 1, 2, 2)),
            ((2, 3, 4, 5), (4, 3, 0, 1)),
        ]
    )
    def test_reflection_pad2d(self, shape, padding):
        class TestModule(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.reflection_pad2d.default(input, padding)

        input = [torch.randn(shape)]
        self.run_test(
            TestModule(),
            input,
        )

    @parameterized.expand(
        [
            (
                "4d",
                (1, 1, 1, 1),
                (2, 2, 2, 2),
                (3, 3, 3, 3),
                torch.float,
                (1, 1, 2, 2),
            ),
        ]
    )
    def test_dynamic_shape_reflection_pad2d(
        self, _, min_shape, opt_shape, max_shape, type, padding
    ):
        class reflection_pad2d(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.reflection_pad2d.default(input, padding)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]

        self.run_test_with_dynamic_shape(reflection_pad2d(), input_specs)

    @parameterized.expand(
        [
            # Per pytorch doc, the input should be 4D or 5D
            ((2, 2, 2, 2), (1, 1, 1, 1, 1, 1)),
            ((1, 2, 3, 4), (3, 2, 2, 1, 1, 1)),
            ((2, 2, 3, 3, 4), (3, 3, 2, 1, 1, 2)),
            ((2, 3, 4, 5, 6), (4, 3, 2, 1, 1, 0)),
        ]
    )
    def test_reflection_pad3d(self, shape, padding):
        class TestModule(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.reflection_pad3d.default(input, padding)

        input = [torch.randn(shape)]
        self.run_test(
            TestModule(),
            input,
        )

    @parameterized.expand(
        [
            (
                "5d",
                (1, 1, 1, 1, 1),
                (2, 2, 2, 2, 2),
                (3, 3, 3, 3, 3),
                torch.float,
                (1, 2, 2, 1, 1, 2),
            ),
        ]
    )
    def test_dynamic_shape_reflection_pad3d(
        self, _, min_shape, opt_shape, max_shape, type, padding
    ):
        class reflection_pad3d(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.reflection_pad3d.default(input, padding)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]

        self.run_test_with_dynamic_shape(reflection_pad3d(), input_specs)


class TestReplicationPadConverter(DispatchTestCase):
    @parameterized.expand(
        [
            # Per pytorch doc, the input should be 2D or 3D
            ((3, 3), (1, 1)),
            ((3, 3), (2, 2)),
            ((2, 2, 2), (1, 1)),
            ((2, 2, 4), (2, 3)),
        ]
    )
    def test_replication_pad1d(self, shape, padding):
        class TestModule(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.replication_pad1d.default(input, padding)

        input = [torch.randn(shape)]
        self.run_test(
            TestModule(),
            input,
        )

    @parameterized.expand(
        [
            # Per pytorch doc, the input should be 3D or 4D
            ((2, 2, 2), (1, 1, 1, 1)),
            ((1, 2, 4), (2, 2, 1, 1)),
            ((2, 2, 3, 3), (1, 1, 2, 2)),
            ((2, 3, 4, 5), (4, 3, 0, 1)),
        ]
    )
    def test_replication_pad2d(self, shape, padding):
        class TestModule(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.replication_pad2d.default(input, padding)

        input = [torch.randn(shape)]
        self.run_test(
            TestModule(),
            input,
        )

    @parameterized.expand(
        [
            # Per pytorch doc, the input should be 4D or 5D
            ((2, 2, 2, 2), (1, 1, 1, 1, 1, 1)),
            ((1, 2, 3, 4), (3, 2, 2, 1, 1, 1)),
            ((2, 2, 3, 3, 4), (3, 3, 2, 1, 1, 2)),
            ((2, 3, 4, 5, 6), (4, 3, 2, 1, 1, 0)),
        ]
    )
    def test_replication_pad3d(self, shape, padding):
        class TestModule(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.replication_pad3d.default(input, padding)

        input = [torch.randn(shape)]
        self.run_test(
            TestModule(),
            input,
        )


class TestCircularPadConverter(DispatchTestCase):
    @parameterized.expand(
        [
            # Per pytorch doc, the input should be 2D or 3D
            ((3, 3), (1, 1)),
            ((3, 3), (2, 2)),
            ((2, 2, 2), (1, 1)),
            ((2, 2, 4), (2, 3)),
        ]
    )
    def test_circular_pad1d(self, shape, pad):
        class TestModule(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten._pad_circular.default(input, pad)

        input = [torch.randn(shape)]
        self.run_test(
            TestModule(),
            input,
        )

    @parameterized.expand(
        [
            # Per pytorch doc, the input should be 3D or 4D
            ((2, 2, 2), (1, 1, 1, 1)),
            ((1, 2, 4), (2, 2, 1, 1)),
            ((2, 2, 3, 3), (1, 1, 2, 2)),
            ((2, 3, 4, 5), (4, 3, 0, 1)),
        ]
    )
    def test_circular_pad2d(self, shape, pad):
        class TestModule(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten._pad_circular.default(input, pad)

        input = [torch.randn(shape)]
        self.run_test(
            TestModule(),
            input,
        )

    @parameterized.expand(
        [
            # Per pytorch doc, the input should be 4D or 5D
            ((2, 2, 2, 2), (1, 1, 1, 1, 1, 1)),
            ((1, 2, 3, 4), (3, 2, 2, 1, 1, 1)),
            ((2, 2, 3, 3, 4), (3, 3, 2, 1, 1, 2)),
            ((2, 3, 4, 5, 6), (4, 3, 2, 1, 1, 0)),
        ]
    )
    def test_circular_pad3d(self, shape, pad):
        class TestModule(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten._pad_circular.default(input, pad)

        input = [torch.randn(shape)]
        self.run_test(
            TestModule(),
            input,
        )


class TestPadConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((3, 3), (2, 2), "constant"),
            ((2, 2, 4), (2, 3, 1, 0), "reflect"),
            ((1, 2, 3, 4), (3, 2, 2, 1, 1, 1), "replicate"),
            ((2, 3, 4, 5), (3, 2, 1, 0), "circular"),
        ]
    )
    def test_pad(self, shape, pad, mode, value=None):
        class TestModule(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.pad.default(input, pad, mode, value)

        input = [torch.randn(shape)]
        self.run_test(
            TestModule(),
            input,
        )


if __name__ == "__main__":
    run_tests()
