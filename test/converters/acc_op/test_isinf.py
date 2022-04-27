import torch
import fx2trt_oss.tracer.acc_tracer.acc_ops as acc_ops
from torch.testing._internal.common_fx2trt import AccTestCase
from torch.testing._internal.common_utils import run_tests

class TestInfConverter(AccTestCase):
    def test_isinf(self):
        class Test(torch.nn.Module):
            def forward(self, x):
                return torch.isinf(x)

        input = torch.randn(2, 3)
        input[0][0] = float('inf')
        input[0][1] = float('-inf')
        input.cuda()
        inputs = [
            input,
        ]
        self.run_test(Test(), inputs, expected_ops={acc_ops.isinf}, test_implicit_batch_dim = False)

    def test_isinf_large(self):
        class Test(torch.nn.Module):
            def forward(self, x):
                return torch.isinf(x)

        input = torch.randn(2, 3, 4, 5)
        input[0][0][0][:]= float('inf')
        input[0][0][1][:] = float('-inf')
        input.cuda()
        inputs = [
            input,
        ]
        self.run_test(Test(), inputs, expected_ops={acc_ops.isinf}, test_implicit_batch_dim = False)

if __name__ == '__main__':
    run_tests()
