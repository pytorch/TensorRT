import torch
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec
from torch_tensorrt.fx.utils import LowerPrecision


class TestToConverter(AccTestCase):
    def test_fp16(self):
        class To(torch.nn.Module):
            def forward(self, x):
                return x.to(torch.float16)

        input = torch.randn(2, 2)
        inputs = [
            input,
        ]
        self.run_test(
            To(),
            inputs,
            expected_ops={acc_ops.to_dtype},
            test_implicit_batch_dim=False,
            precision=LowerPrecision.FP16,
        )

    # Testing with shape shape=(-1, -1, -1, -1) results into following error:
    # Error: assert engine
    """
    def test_fp16_with_dynamic_shape_four_dimension(self):
        class To(torch.nn.Module):
            def forward(self, x):
                return x.to(torch.float16)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float16,
                shape_ranges=[((1, 1, 3, 3), (3, 3, 3, 3), (3, 3, 3, 3))],
            ).cuda(),
        ]

        self.run_test_with_dynamic_shape(
            To(), input_specs, expected_ops={acc_ops.to_dtype}
        )
    """

    def test_fp32(self):
        class To(torch.nn.Module):
            def forward(self, x):
                return x.to(torch.float32)

        input = torch.randn(2, 2).to(torch.float16)
        inputs = [
            input,
        ]
        self.run_test(
            To(), inputs, expected_ops={acc_ops.to_dtype}, test_implicit_batch_dim=False
        )

    def test_cuda_fp16(self):
        class To(torch.nn.Module):
            def forward(self, x):
                return x.to(torch.device("cuda:0"), torch.float16)

        input = torch.randn(2, 2)
        inputs = [
            input,
        ]
        self.run_test(
            To(),
            inputs,
            expected_ops={acc_ops.to_dtype},
            test_implicit_batch_dim=False,
            precision=LowerPrecision.FP16,
        )

    def test_cuda(self):
        class To(torch.nn.Module):
            def forward(self, x):
                x = x.to(torch.device("cuda"))
                # append extra layer since to(device) is skipped in TRT
                return x + torch.randn(2, 2).cuda()

        input = torch.randn(2, 2)
        inputs = [
            input,
        ]
        self.run_test(
            To(),
            inputs,
            expected_ops={acc_ops.to_dtype, acc_ops.add},
            test_implicit_batch_dim=False,
            precision=LowerPrecision.FP32,
        )

    def test_cuda_with_dynamic_shape_four_dimensions(self):
        class To(torch.nn.Module):
            def forward(self, x):
                x = x.to(torch.device("cuda"))
                # append extra layer since to(device) is skipped in TRT
                return x + torch.randn(3, 3, 3, 3).cuda()

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float16,
                shape_ranges=[((1, 1, 3, 3), (3, 3, 3, 3), (3, 3, 3, 3))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            To(), input_specs, expected_ops={acc_ops.to_dtype, acc_ops.add}
        )

    def test_device(self):
        class To(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.randn(2, 2)

            def forward(self, x):
                idevice = x.device
                a = self.a.to(idevice)
                return x + a

        input = torch.randn(2, 2).cuda()
        inputs = [
            input,
        ]
        self.run_test(
            To(),
            inputs,
            expected_ops={acc_ops.to_dtype},
            test_implicit_batch_dim=False,
            precision=LowerPrecision.FP32,
        )

    def test_device_with_dynamic_shape_four_dimensions(self):
        class To(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.randn(3, 3, 3, 3)

            def forward(self, x):
                idevice = x.device
                a = self.a.to(idevice)
                return x + a

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float16,
                shape_ranges=[((1, 1, 3, 3), (3, 3, 3, 3), (3, 3, 3, 3))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            To(), input_specs, expected_ops={acc_ops.to_dtype, acc_ops.add}
        )

    def test_device_fp16(self):
        class To(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.randn(2, 2)

            def forward(self, x):
                idevice = x.device
                idtype = x.dtype
                a = self.a.to(idevice)
                # fx tracer could not handle "to(idevice, torch.float16)"
                # TypeError: to() received an invalid combination of arguments - got (Attribute, torch.dtype)
                return a.to(idtype)

        input = torch.randn(2, 2).half().cuda()
        inputs = [
            input,
        ]
        self.run_test(
            To(),
            inputs,
            expected_ops={acc_ops.to_dtype},
            test_implicit_batch_dim=False,
            precision=LowerPrecision.FP16,
        )

    # Testing with shape shape=(-1, -1, -1, -1) results into following error:
    # Error: assert engine
    """
    def test_device_fp16_with_dynamic_shape_four_dimensions(self):
        class To(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.randn(2, 2)

            def forward(self, x):
                idevice = x.device
                idtype = x.dtype
                a = self.a.to(idevice)
                # fx tracer could not handle "to(idevice, torch.float16)"
                # TypeError: to() received an invalid combination of arguments - got (Attribute, torch.dtype)
                return a.to(idtype)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float16,
                shape_ranges=[((2, 2, 2, 2), (4, 4, 4, 4), (4, 4, 4, 4))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            To(), input_specs, expected_ops={acc_ops.to_dtype}
        )
    """

    # tensor.float()
    def test_float(self):
        class To(torch.nn.Module):
            def forward(self, x):
                return x.float()

        input = torch.randn(2, 2).half()
        inputs = [
            input,
        ]
        self.run_test(
            To(),
            inputs,
            expected_ops={acc_ops.to_dtype},
            test_implicit_batch_dim=False,
            precision=LowerPrecision.FP32,
        )

    # tensor.float()
    def test_float_with_dynamic_shape_four_dimensions(self):
        class To(torch.nn.Module):
            def forward(self, x):
                return x.float()

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (3, 3, 3, 3), (3, 3, 3, 3))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            To(), input_specs, expected_ops={acc_ops.to_dtype}
        )

    # Half is not suitable for dynamic shape
    # Error: assert engine

    # tensor.half()
    def test_half(self):
        class To(torch.nn.Module):
            def forward(self, x):
                return x.half()

        input = torch.randn(2, 2)
        inputs = [
            input,
        ]
        self.run_test(
            To(),
            inputs,
            expected_ops={acc_ops.to_dtype},
            test_implicit_batch_dim=False,
            precision=LowerPrecision.FP16,
        )

    # tensor.int()
    def test_int(self):
        class To(torch.nn.Module):
            def forward(self, x):
                x = x.int()
                # we do not expect int to be output type, so add an extra layer
                x = x.float()
                return x

        input = torch.randn(2, 2)
        inputs = [
            input,
        ]
        self.run_test(
            To(),
            inputs,
            expected_ops={acc_ops.to_dtype},
            test_implicit_batch_dim=False,
            precision=LowerPrecision.FP32,
        )

    # tensor.int()
    def test_int_with_dynamic_shape_four_dimensions(self):
        class To(torch.nn.Module):
            def forward(self, x):
                x = x.int()
                # we do not expect int to be output type, so add an extra layer
                x = x.float()
                return x

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.int,
                shape_ranges=[((1, 1, 1, 1), (3, 3, 3, 3), (3, 3, 3, 3))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            To(), input_specs, expected_ops={acc_ops.to_dtype}
        )


if __name__ == "__main__":
    run_tests()
