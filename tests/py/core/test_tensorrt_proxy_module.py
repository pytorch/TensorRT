# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import io
import sys
import unittest

import torch_tensorrt  # noqa: F401
from torch_tensorrt._TensorRTProxyModule import TensorRTProxyModule


class TestTensorRTProxyModule(unittest.TestCase):
    def test_missing_attribute_prints_nothing(self):
        trt = sys.modules["tensorrt"]
        self.assertIsInstance(trt, TensorRTProxyModule)

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertFalse(hasattr(trt, "no_such_attribute"))
        self.assertEqual(stdout.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
