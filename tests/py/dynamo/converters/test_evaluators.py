import operator
import unittest

import torch
import torch.nn as nn
from harness import DispatchTestCase
from torch.testing._internal.common_utils import run_tests


class TestCloneConverter(DispatchTestCase):
    def test_clone_contiguous(self):
        class Clone(nn.Module):
            def forward(self, x):
                y = torch.clone(x, memory_format=torch.contiguous_format)
                return y + 1

        inputs = [torch.randn((1, 3, 10))]
        self.run_test(
            Clone(),
            inputs,
            expected_ops={torch.ops.aten.clone.default},
            disable_passes=True,
        )

    def test_clone_regular(self):
        class Clone(nn.Module):
            def forward(self, x):
                y = torch.clone(x)
                return y + 1

        inputs = [torch.randn((8, 2, 10))]
        self.run_test(
            Clone(),
            inputs,
            expected_ops={torch.ops.aten.clone.default},
            disable_passes=True,
        )


# TODO: Switch this test back to self.run_test once an implementation exists
# for a converter that returns a list, such as aten.split
@unittest.skip("Pending aten.split converter. Currently tested by E2E")
class TestGetItemConverter(DispatchTestCase):
    def test_getitem(self):
        class GetItem(nn.Module):
            def forward(self, x):
                lis = torch.split(x, 5)
                b = operator.getitem(lis, 0)
                c = operator.getitem(lis, 1)
                d = b + c
                return d

        inputs = [
            torch.randn((3, 3, 10)),
            torch.randn((3, 3, 10)),
            torch.randn((3, 3, 10)),
        ]
        self.run_test(
            GetItem(),
            inputs,
            expected_ops={operator.getitem},
            disable_passes=True,
        )


if __name__ == "__main__":
    run_tests()
