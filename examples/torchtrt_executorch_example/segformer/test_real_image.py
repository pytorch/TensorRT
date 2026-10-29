# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-only checks for the numerical/visualization helper (requires Torch)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from validate_real_image import LABELS, PALETTE, compare, prediction, read_logits


class RealImageTests(unittest.TestCase):
    def test_class_order(self):
        self.assertEqual(PALETTE.shape, (19, 3))
        self.assertEqual(LABELS[0], "road")
        self.assertEqual(LABELS[13], "car")

    def test_read_exact_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logits.bin"
            array = np.arange(19 * 4, dtype=np.float32).reshape(1, 19, 2, 2)
            array.tofile(path)
            np.testing.assert_array_equal(
                read_logits(path, [1, 19, 2, 2]).numpy(), array
            )

    def test_reject_truncated_or_extra_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logits.bin"
            for count in (19 * 4 - 1, 19 * 4 + 1):
                np.zeros(count, dtype=np.float32).tofile(path)
                with self.assertRaisesRegex(ValueError, "byte count"):
                    read_logits(path, [1, 19, 2, 2])

    def test_reject_bad_shape_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logits.bin"
            np.full(19 * 4, np.nan, dtype=np.float32).tofile(path)
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                read_logits(path, [1, 19, 2, 2])
            with self.assertRaisesRegex(ValueError, "1x19xHxW"):
                read_logits(path, [19, 2, 2])

    def test_comparison_rejects_changed_logits(self):
        reference = torch.zeros(1, 19, 2, 2)
        self.assertEqual(compare(reference, reference)["mismatches"], 0)
        changed = reference.clone()
        changed[0, 0, 0, 0] = 0.1
        with self.assertRaises(AssertionError):
            compare(changed, reference)

    def test_postprocess_interpolates_logits_not_ids(self):
        logits = torch.full((1, 19, 1, 2), -100.0)
        logits[0, 0] = torch.tensor([10.0, 0.0])
        logits[0, 1] = torch.tensor([0.0, 1.0])
        # Class 0 still wins at pixel 2 after bilinear logit interpolation;
        # nearest-resizing the two source class IDs would incorrectly give 1.
        self.assertEqual(prediction(logits, (1, 4)).tolist(), [[0, 0, 0, 1]])


if __name__ == "__main__":
    unittest.main()
