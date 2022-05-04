import fx2trt_oss.tracer.acc_tracer.acc_ops as acc_ops
import torch
from fx2trt_oss.fx.utils import LowerPrecision
from torch.testing._internal.common_fx2trt import AccTestCase
from torch.testing._internal.common_utils import run_tests


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

    def test_tensor(self):
        class To(torch.nn.Module):
            def forward(self, x, y):
                return y.to(x)

        input = torch.randn(2, 2).half().cuda()
        other = torch.randn(2, 2)
        inputs = [
            input,
            other,
        ]
        self.run_test(
            To(),
            inputs,
            expected_ops={acc_ops.to_dtype},
            test_implicit_batch_dim=False,
            precision=LowerPrecision.FP16,
        )

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


if __name__ == "__main__":
    run_tests()
