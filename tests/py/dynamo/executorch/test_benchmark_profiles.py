# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-only protocol checks; not model inference tests."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

# Resolve example helpers from this checkout even when CI runs from tests/py/dynamo.
with patch.object(sys, "path", [str(Path(__file__).resolve().parents[4]), *sys.path]):
    from examples.torchtrt_executorch_example.native_validation.benchmarks.benchmark_python import (
        INPUT_SHAPES,
        model_spec,
    )

pytestmark = pytest.mark.unit


class ProfileTests(unittest.TestCase):
    def test_original_bevformer_protocol_unchanged(self):
        self.assertEqual(model_spec("bevformer"), (INPUT_SHAPES, 3, "model.py"))

    def test_segformer_static_shape(self):
        self.assertEqual(
            model_spec("segformer"), (((1, 3, 1024, 1024),), 1, "export_full.py")
        )

    def test_unknown_profile_is_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            model_spec("unverified")


if __name__ == "__main__":
    unittest.main()
