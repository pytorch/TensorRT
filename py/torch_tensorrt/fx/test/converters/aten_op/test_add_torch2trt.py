import torch
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import DispatchTestCase, InputTensorSpec

class TestAddBasic(DispatchTestCase):
    def test_add_basic(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()

            def forward(self, x, y):
                return x + y
        
        inputs = [torch.randn(1,3,224,224), torch.randn(1,3,224,224)]
        print("============Running test before=============")
        self.run_test(
            TestModule(),
            inputs,
            expected_ops=({torch.ops.aten.add.Tensor}),
        )

if __name__ == "__main__":
    run_tests()

#@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
#def test_add_basic():
#return Add()