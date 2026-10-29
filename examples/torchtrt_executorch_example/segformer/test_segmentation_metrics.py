# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hand-computed metric and dataset-selection tests; NOT model accuracy results."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from evaluate_cityscapes import pairs, validate_manifest
from segmentation_metrics import LABEL_IDS, confusion_matrix, encode_mask, summarize


class SegmentationMetricsTests(unittest.TestCase):
    def test_all_cityscapes_evaluated_ids(self):
        mask = np.array([LABEL_IDS], dtype=np.uint8)
        np.testing.assert_array_equal(
            encode_mask(mask, "label_ids"), np.arange(19)[None]
        )

    def test_void_and_license_plate_are_ignored(self):
        mask = np.array([[0, 1, 6, 9, 10, 14, 15, 16, 18, 29, 30, 255]], dtype=np.uint8)
        self.assertTrue((encode_mask(mask, "label_ids") == 255).all())

    def test_unknown_labels_rejected(self):
        for encoding, value in (
            ("label_ids", 34),
            ("train_ids", 19),
            ("train_ids", -1),
        ):
            with self.subTest(encoding=encoding, value=value), self.assertRaises(
                ValueError
            ):
                encode_mask(np.array([[value]], dtype=np.int32), encoding)

    def test_color_or_float_masks_rejected(self):
        for mask in (
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.zeros((2, 2), dtype=np.float32),
        ):
            with self.assertRaises(ValueError):
                encode_mask(mask, "label_ids")

    def test_hand_computed_iou_and_pixel_accuracy(self):
        truth = np.array([[0, 0, 1, 1, 255]])
        pred = np.array([[0, 1, 1, 1, 18]])
        matrix = confusion_matrix(truth, pred)
        self.assertEqual(matrix[:2, :2].tolist(), [[1, 1], [0, 2]])
        result = summarize(matrix)
        self.assertEqual(result["valid_pixels"], 4)
        self.assertAlmostEqual(result["pixel_accuracy"], 3 / 4)
        self.assertAlmostEqual(result["per_class"]["road"]["iou"], 1 / 2)
        self.assertAlmostEqual(result["per_class"]["sidewalk"]["iou"], 2 / 3)
        self.assertAlmostEqual(result["mean_iou"], (1 / 2 + 2 / 3) / 2)
        self.assertIsNone(result["per_class"]["bicycle"]["iou"])

    def test_false_positive_only_class_is_not_excluded(self):
        result = summarize(confusion_matrix(np.array([[0, 0]]), np.array([[0, 2]])))
        self.assertEqual(result["per_class"]["building"]["iou"], 0)
        self.assertEqual(result["evaluated_classes"], 2)
        self.assertEqual(result["mean_iou"], 0.25)

    def test_aggregate_counts_not_per_image_average(self):
        a = confusion_matrix(np.zeros((1, 1), dtype=int), np.zeros((1, 1), dtype=int))
        b = confusion_matrix(np.zeros((1, 9), dtype=int), np.ones((1, 9), dtype=int))
        result = summarize(a + b)
        self.assertEqual(result["pixel_accuracy"], 0.1)
        self.assertEqual(result["mean_iou"], 0.05)

    def test_all_ignored_does_not_create_accuracy(self):
        result = summarize(confusion_matrix(np.array([[255]]), np.array([[0]])))
        self.assertIsNone(result["mean_iou"])
        self.assertIsNone(result["pixel_accuracy"])
        json.dumps(result, allow_nan=False)

    def test_bad_prediction_or_confusion_matrix_rejected(self):
        for prediction in (np.array([[255]]), np.array([[0.0]]), np.array([0])):
            with self.assertRaises(ValueError):
                confusion_matrix(np.array([[0]]), prediction)
        with self.assertRaises(ValueError):
            summarize(np.full((19, 19), -1))


class DatasetSelectionTests(unittest.TestCase):
    def make_dataset_paths(self, root):
        # Empty filenames suffice for selection tests; no real dataset is used.
        for index in range(12):
            image = (
                root / f"leftImg8bit/val/test/test_000000_{index:06d}_leftImg8bit.png"
            )
            mask = root / f"gtFine/val/test/test_000000_{index:06d}_gtFine_labelIds.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            mask.parent.mkdir(parents=True, exist_ok=True)
            image.touch()
            mask.touch()

    def test_selection_is_repeatable_and_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_dataset_paths(root)
            a, available = pairs(root, "label_ids", 5, 0)
            b, _ = pairs(root, "label_ids", 5, 0)
            self.assertEqual(a, b)
            self.assertEqual(available, 12)
            self.assertEqual(len(pairs(root, "label_ids", 0, 0)[0]), 12)

    def test_missing_image_or_mask_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                pairs(root, "label_ids", 0, 0)
            self.make_dataset_paths(root)
            with self.assertRaises(FileNotFoundError):
                pairs(root, "train_ids", 0, 0)

    def test_invalid_limits_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_dataset_paths(root)
            for limit in (-1, 13):
                with self.assertRaises(ValueError):
                    pairs(root, "label_ids", limit, 0)

    def test_incomplete_or_unsafe_manifest_fails(self):
        for ids in ([], ["a", "a"], ["../x"]):
            with self.assertRaises(ValueError):
                validate_manifest(
                    {
                        "schema": 1,
                        "selected_images": len(ids),
                        "cases": [{"id": x} for x in ids],
                    }
                )
        validate_manifest(
            {"schema": 1, "selected_images": 1, "cases": [{"id": "city_1"}]}
        )


if __name__ == "__main__":
    unittest.main()
