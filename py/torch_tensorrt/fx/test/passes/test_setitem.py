import torch
import torchdynamo
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.passes.lower_basic_pass import transform_setitem
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase
from torchdynamo.optimizations import backends


class TestTransformSetitem(AccTestCase):
    def test_setitem1d(self):
        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                y[0:2] = x
                return y

        inputs = [torch.randn(2), torch.randn(3)]
        m = TestModule()

        inputs = [i.cuda() for i in inputs]
        m.cuda()

        def transform_fx(gm, example_inputs):
            gm = transform_setitem(gm, example_inputs)
            return gm

        optimize_ctx = torchdynamo.optimize(
            transform_fx,
            nopython=True,
        )

        with optimize_ctx:
            m(*inputs)

    def test_setitem1d_c2(self):
        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                y[:-1] = x
                y[1:] = x
                return y

        inputs = [torch.randn(2), torch.randn(3)]
        m = TestModule()

        inputs = [i.cuda() for i in inputs]
        m.cuda()

        def transform_fx(gm, example_inputs):
            gm = transform_setitem(gm, example_inputs)
            return gm

        optimize_ctx = torchdynamo.optimize(
            transform_fx,
            nopython=True,
        )

        with optimize_ctx:
            m(*inputs)

    def test_setitem1d_c3(self):
        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                y[1] = x
                return y

        inputs = [torch.randn(2), torch.randn(3)]
        m = TestModule()

        inputs = [i.cuda() for i in inputs]
        m.cuda()

        def transform_fx(gm, example_inputs):
            gm = transform_setitem(gm, example_inputs)
            return gm

        optimize_ctx = torchdynamo.optimize(
            transform_fx,
            nopython=True,
        )

        with optimize_ctx:
            m(*inputs)

    @parameterized.expand(
        [
            ("c1", (4, 2), (4, 5), 0, 2),
            ("c2", (4, 2), (4, 5), 1, 3),
        ]
    )
    def test_setitem2d_1v(self, name, x_shape, y_shape, y_start, y_end):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                y[:, y_start:y_end] = x
                return y

        inputs = [torch.randn(x_shape), torch.randn(y_shape)]
        m = TestModule()

        inputs = [i.cuda() for i in inputs]
        m.cuda()

        def transform_fx(gm, example_inputs):
            gm = transform_setitem(gm, example_inputs)
            return gm

        optimize_ctx = torchdynamo.optimize(
            transform_fx,
            nopython=True,
        )
        with optimize_ctx:
            m(*inputs)

    @parameterized.expand(
        [
            ("c1", (4, 2), (8, 2), 0, 2),
            ("c2", (4, 2), (8, 2), 1, 3),
        ]
    )
    def test_setitem2d_1v_ex(self, name, x_shape, y_shape, y_start, y_end):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                y[y_start:y_end, :] = x
                return y

        inputs = [torch.randn(x_shape), torch.randn(y_shape)]
        m = TestModule()

        inputs = [i.cuda() for i in inputs]
        m.cuda()

        def transform_fx(gm, example_inputs):
            gm = transform_setitem(gm, example_inputs)
            return gm

        optimize_ctx = torchdynamo.optimize(
            transform_fx,
            nopython=True,
        )
        with optimize_ctx:
            m(*inputs)

    @parameterized.expand(
        [
            ("c1", (4, 2), (4, 2), 0, 1),
        ]
    )
    def test_setitem2d_1v_ex2(self, name, x_shape, y_shape, y_start, y_end):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                y[:, y_start:y_end] = x[:, 0]
                return y

        inputs = [torch.randn(x_shape), torch.randn(y_shape)]
        m = TestModule()

        inputs = [i.cuda() for i in inputs]
        m.cuda()

        def transform_fx(gm, example_inputs):
            gm = transform_setitem(gm, example_inputs)
            return gm

        optimize_ctx = torchdynamo.optimize(
            transform_fx,
            nopython=True,
        )
        with optimize_ctx:
            m(*inputs)

    @parameterized.expand(
        [
            ("c1", (3, 2), (4, 5), 0, 3, 0, 2),
            ("c2", (3, 2), (4, 5), 1, 4, 1, 3),
        ]
    )
    def test_setitem2d_2v(self, name, x_shape, y_shape, x_start, x_end, y_start, y_end):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                y[x_start:x_end, y_start:y_end] = x
                y = y + 3
                return y

        inputs = [torch.randn(x_shape), torch.randn(y_shape)]
        m = TestModule()

        inputs = [i.cuda() for i in inputs]
        m.cuda()

        def transform_fx(gm, example_inputs):
            gm = transform_setitem(gm, example_inputs)
            return gm

        optimize_ctx = torchdynamo.optimize(
            transform_fx,
            nopython=True,
        )
        with optimize_ctx:
            m(*inputs)

    @parameterized.expand(
        [
            ("c1", (2, 3, 4), (2, 5, 6), 0, 3, 0, 4),
            ("c2", (2, 3, 4), (2, 5, 6), 1, 4, 1, 5),
        ]
    )
    def test_setitem3d_2v(self, name, x_shape, y_shape, start_1, end_1, start_2, end_2):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                y[:, start_1:end_1, start_2:end_2] = x
                y = y + 3
                return y

        inputs = [torch.randn(x_shape), torch.randn(y_shape)]
        m = TestModule()

        inputs = [i.cuda() for i in inputs]
        m.cuda()

        def transform_fx(gm, example_inputs):
            gm = transform_setitem(gm, example_inputs)
            return gm

        optimize_ctx = torchdynamo.optimize(
            transform_fx,
            nopython=True,
        )
        with optimize_ctx:
            m(*inputs)

    @parameterized.expand(
        [
            ("c1", (3, 2, 4), (5, 2, 6), 0, 3, 0, 4),
            ("c2", (3, 2, 4), (5, 2, 6), 1, 4, 1, 5),
        ]
    )
    def test_setitem3d_2v_ext(
        self, name, x_shape, y_shape, start_0, end_0, start_2, end_2
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                y[start_0:end_0, :, start_2:end_2] = x
                y = y + 3
                return y

        inputs = [torch.randn(x_shape), torch.randn(y_shape)]
        m = TestModule()

        inputs = [i.cuda() for i in inputs]
        m.cuda()

        def transform_fx(gm, example_inputs):
            gm = transform_setitem(gm, example_inputs)
            return gm

        optimize_ctx = torchdynamo.optimize(
            transform_fx,
            nopython=True,
        )
        with optimize_ctx:
            m(*inputs)

    @parameterized.expand(
        [
            ("c1", (2, 3, 4), (4, 5, 6), 0, 2, 0, 3, 0, 4),
            ("c2", (2, 3, 4), (4, 5, 6), 1, 3, 1, 4, 1, 5),
        ]
    )
    def test_setitem3d_3v(
        self, name, x_shape, y_shape, start_0, end_0, start_1, end_1, start_2, end_2
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                y[start_0:end_0, start_1:end_1, start_2:end_2] = x
                y = y + 3
                x = y[start_0:end_0, start_1:end_1, start_2:end_2]
                return x

        inputs = [torch.randn(x_shape), torch.randn(y_shape)]
        m = TestModule()

        inputs = [i.cuda() for i in inputs]
        m.cuda()

        def transform_fx(gm, example_inputs):
            gm = transform_setitem(gm, example_inputs)
            return gm

        optimize_ctx = torchdynamo.optimize(
            transform_fx,
            nopython=True,
        )
        with optimize_ctx:
            m(*inputs)

    @parameterized.expand(
        [
            ("c1", (2, 3, 4, 5), (2, 3, 6, 7), 0, 4, 0, 5),
            ("c2", (2, 3, 4, 5), (2, 3, 6, 7), 1, 5, 1, 6),
        ]
    )
    def test_setitem4d_2v(self, name, x_shape, y_shape, start_2, end_2, start_3, end_3):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                y[:, :, start_2:end_2, start_3:end_3] = x
                y = y + 3
                return y

        inputs = [torch.randn(x_shape), torch.randn(y_shape)]
        m = TestModule()

        inputs = [i.cuda() for i in inputs]
        m.cuda()

        def transform_fx(gm, example_inputs):
            gm = transform_setitem(gm, example_inputs)
            return gm

        optimize_ctx = torchdynamo.optimize(
            transform_fx,
            nopython=True,
        )
        with optimize_ctx:
            m(*inputs)

    @parameterized.expand(
        [
            ("c1", (2, 3, 4, 5), (2, 5, 4, 7), 0, 3, 0, 5),
            ("c2", (2, 3, 4, 5), (2, 5, 4, 7), 1, 4, 1, 6),
        ]
    )
    def test_setitem4d_2v_ext(
        self, name, x_shape, y_shape, start_1, end_1, start_3, end_3
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                y[:, start_1:end_1, :, start_3:end_3] = x
                y = y + 3
                return y

        inputs = [torch.randn(x_shape), torch.randn(y_shape)]
        m = TestModule()

        inputs = [i.cuda() for i in inputs]
        m.cuda()

        def transform_fx(gm, example_inputs):
            gm = transform_setitem(gm, example_inputs)
            return gm

        optimize_ctx = torchdynamo.optimize(
            transform_fx,
            nopython=True,
        )
        with optimize_ctx:
            m(*inputs)

    @parameterized.expand(
        [
            ("c1", (2, 3, 4, 5), (2, 5, 6, 7), 0, 3, 0, 4, 0, 5),
            ("c2", (2, 3, 4, 5), (2, 5, 6, 7), 1, 4, 1, 5, 1, 6),
        ]
    )
    def test_setitem4d_3v(
        self, name, x_shape, y_shape, start_1, end_1, start_2, end_2, start_3, end_3
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                y[:, start_1:end_1, start_2:end_2, start_3:end_3] = x
                y = y + 3
                return y

        inputs = [torch.randn(x_shape), torch.randn(y_shape)]
        m = TestModule()

        inputs = [i.cuda() for i in inputs]
        m.cuda()

        def transform_fx(gm, example_inputs):
            gm = transform_setitem(gm, example_inputs)
            return gm

        optimize_ctx = torchdynamo.optimize(
            transform_fx,
            nopython=True,
        )
        with optimize_ctx:
            m(*inputs)

    @parameterized.expand(
        [
            ("c1", (2, 3, 4, 5), (4, 5, 6, 7), 0, 2, 0, 3, 0, 4, 0, 5),
            ("c2", (2, 3, 4, 5), (4, 5, 6, 7), 1, 3, 1, 4, 1, 5, 1, 6),
        ]
    )
    def test_setitem4d_4v(
        self,
        name,
        x_shape,
        y_shape,
        start_0,
        end_0,
        start_1,
        end_1,
        start_2,
        end_2,
        start_3,
        end_3,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                y[start_0:end_0, start_1:end_1, start_2:end_2, start_3:end_3] = x
                y = y + 3
                x = y[start_0:end_0, start_1:end_1, start_2:end_2, start_3:end_3]
                return x

        inputs = [torch.randn(x_shape), torch.randn(y_shape)]
        m = TestModule()

        inputs = [i.cuda() for i in inputs]
        m.cuda()

        def transform_fx(gm, example_inputs):
            gm = transform_setitem(gm, example_inputs)
            return gm

        optimize_ctx = torchdynamo.optimize(
            transform_fx,
            nopython=True,
        )
        with optimize_ctx:
            m(*inputs)

    @parameterized.expand(
        [
            ("c1", (2, 3, 4, 5, 6), (4, 5, 6, 7, 6), 0, 2, 0, 3, 0, 4, 0, 5),
        ]
    )
    def test_setitem5d_warning(
        self,
        name,
        x_shape,
        y_shape,
        start_0,
        end_0,
        start_1,
        end_1,
        start_2,
        end_2,
        start_3,
        end_3,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                y[start_0:end_0, start_1:end_1, start_2:end_2, start_3:end_3, :] = x
                y = y + 3
                x = y[start_0:end_0, start_1:end_1, start_2:end_2, start_3:end_3]
                return x

        inputs = [torch.randn(x_shape), torch.randn(y_shape)]
        m = TestModule()

        inputs = [i.cuda() for i in inputs]
        m.cuda()

        def transform_fx(gm, example_inputs):
            gm = transform_setitem(gm, example_inputs)
            return gm

        optimize_ctx = torchdynamo.optimize(
            transform_fx,
            nopython=True,
        )
        with optimize_ctx:
            m(*inputs)

    # test with torchdynamo
    def test_setitem1d_trt(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                y[1] = x
                return y

        inputs = [torch.randn(1), torch.randn(3)]
        m = TestModule()

        inputs = [i.cuda() for i in inputs]
        m.cuda()
        ref_output = m(*inputs)

        optimize_ctx = torchdynamo.optimize(backends.fx2trt_compiler, nopython=True)
        with optimize_ctx:
            output = m(*inputs)
        self.assertTrue(torch.allclose(ref_output, output))

    @parameterized.expand(
        [
            ("c1", (4, 2), (4, 5), 0, 2),
            ("c2", (4, 2), (4, 5), 1, 3),
        ]
    )
    def test_setitem2d_1v_trt(self, name, x_shape, y_shape, y_start, y_end):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                y[:, y_start:y_end] = x
                return y

        inputs = [torch.randn(x_shape), torch.randn(y_shape)]
        m = TestModule()

        inputs = [i.cuda() for i in inputs]
        m.cuda()

        ref_output = m(*inputs)
        optimize_ctx = torchdynamo.optimize(backends.fx2trt_compiler, nopython=True)
        with optimize_ctx:
            output = m(*inputs)
        self.assertTrue(torch.allclose(ref_output, output))

    @parameterized.expand(
        [
            ("c1", (2, 3, 4, 5), (4, 5, 6, 7), 0, 2, 0, 3, 0, 4, 0, 5),
            ("c2", (2, 3, 4, 5), (4, 5, 6, 7), 1, 3, 1, 4, 1, 5, 1, 6),
        ]
    )
    def test_setitem4d_4v_trt(
        self,
        name,
        x_shape,
        y_shape,
        start_0,
        end_0,
        start_1,
        end_1,
        start_2,
        end_2,
        start_3,
        end_3,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                y[start_0:end_0, start_1:end_1, start_2:end_2, start_3:end_3] = x
                y = y + 3
                x = y[start_0:end_0, start_1:end_1, start_2:end_2, start_3:end_3]
                return x

        inputs = [torch.randn(x_shape), torch.randn(y_shape)]
        m = TestModule()

        inputs = [i.cuda() for i in inputs]
        m.cuda()

        ref_output = m(*inputs)
        optimize_ctx = torchdynamo.optimize(backends.fx2trt_compiler, nopython=True)
        with optimize_ctx:
            output = m(*inputs)
        self.assertTrue(torch.allclose(ref_output, output))


if __name__ == "__main__":
    run_tests()
