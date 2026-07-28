#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

# Owner(s): ["oncall: gpu_enablement"]

# Test that this import should not trigger any error when run
# in non-GPU hosts, or in any build mode.
import torch_tensorrt.fx.lower as fxl  # noqa: F401
from torch.testing._internal.common_utils import run_tests, TestCase


class MainTests(TestCase):
    def test_1(self):
        pass


if __name__ == "__main__":
    run_tests()
