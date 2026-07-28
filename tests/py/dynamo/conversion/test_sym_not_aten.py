import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestSymNotConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (torch.tensor(True),),
            (torch.tensor(False),),
            (torch.tensor([True]),),
            (torch.tensor([[True]]),),
            (torch.tensor([[False]]),),
        ]
    )
    def test_sym_not_bool(self, data):
        class sym_not(nn.Module):
            def forward(self, input):
                return torch.sym_not(input)

        inputs = [data]

        self.run_test(
            sym_not(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
