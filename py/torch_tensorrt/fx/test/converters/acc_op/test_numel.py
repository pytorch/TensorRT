import torch
import torch.nn as nn
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase


class TestNumelConverter(AccTestCase):
    def test_numel(self):
        class Numel(nn.Module):
            def forward(self, x):
                return torch.numel(x) * x

        inputs = [torch.ones(1, 2, 3, 4)]
        self.run_test(Numel(), inputs, expected_ops={acc_ops.numel})


if __name__ == "__main__":
    run_tests()
