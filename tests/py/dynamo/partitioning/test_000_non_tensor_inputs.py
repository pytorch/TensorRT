import unittest

import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests


class TestNonTensorInputs(TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_full_compilation_with_bool_kwarg(self):
        class WithReturnDict(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(64, 64)

            def forward(self, hidden_states, return_dict=True):
                out = torch.relu(self.linear(hidden_states))
                if not return_dict:
                    return (out,)
                return {"sample": out}

        model = WithReturnDict().eval().cuda()
        inputs = {
            "hidden_states": torch.randn(8, 64, device="cuda"),
            "return_dict": False,
        }
        exported_program = torch.export.export(model, args=(), kwargs=inputs)
        trt_gm = torch_tensorrt.dynamo.compile(
            exported_program,
            inputs=inputs,
            min_block_size=1,
            require_full_compilation=True,
        )

        torch.testing.assert_close(
            trt_gm(**inputs)[0], model(**inputs)[0], rtol=1e-2, atol=1e-2
        )


if __name__ == "__main__":
    run_tests()
