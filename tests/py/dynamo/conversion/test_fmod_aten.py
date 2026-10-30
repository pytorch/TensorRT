# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase

# In float32, 4.2 / 1.4 and 0.7 / 0.1 round up to whole numbers (3.0 and 7.0), one
# more than the truncated quotient, which gave fmod(4.2, 1.4) = -1.2e-07 instead of 1.4.
LHS = [4.2, -4.2, 4.2, -4.2, 0.7, -0.7, 5.0, -7.5]
RHS = [1.4, 1.4, -1.4, -1.4, 0.1, 0.1, 3.0, 2.5]


class TestFmodConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("fmod", torch.ops.aten.fmod.Tensor),
            ("remainder", torch.ops.aten.remainder.Tensor),
        ]
    )
    def test_quotient_rounds_up(self, _, op):
        class Mod(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return op(lhs_val, rhs_val)

        inputs = [torch.tensor(LHS), torch.tensor(RHS)]
        self.run_test(Mod(), inputs)

    @parameterized.expand(
        [
            ("fmod", torch.ops.aten.fmod.Scalar),
            ("remainder", torch.ops.aten.remainder.Scalar),
        ]
    )
    def test_quotient_rounds_up_scalar(self, _, op):
        class Mod(nn.Module):
            def forward(self, lhs_val):
                return op(lhs_val, 1.4)

        inputs = [torch.tensor([4.2, -4.2, 2.8, -2.8, 0.7, 1.5])]
        self.run_test(Mod(), inputs)

    def test_quotient_rounds_up_int(self):
        class Mod(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.fmod.Tensor(lhs_val, rhs_val)

        inputs = [
            torch.tensor([7, -7, 7, -7, 0], dtype=torch.int32),
            torch.tensor([3, 3, -3, -3, 5], dtype=torch.int32),
        ]
        self.run_test(Mod(), inputs)


if __name__ == "__main__":
    run_tests()
