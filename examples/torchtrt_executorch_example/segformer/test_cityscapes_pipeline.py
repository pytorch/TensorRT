# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Synthetic plumbing tests, NOT a Cityscapes or deployed-model accuracy run.

Uses tiny invented RGB/masks and a fake CPU model. Native outputs/maps are
test doubles. Real native inference is still tested separately by the runner.
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from evaluate_cityscapes import prepare, score
from fetch_checkpoint import MODEL_ID, REVISION, WEIGHTS_SHA256, checksum
from PIL import Image
from segmentation_metrics import LABELS


class FakeModel:
    def cuda(self):
        return self

    def __call__(self, inputs):
        result = torch.zeros(1, 19, 8, 8)
        result[:, 0] = 1  # Always predict road, deliberately imperfect.
        return (result,)


class CityscapesPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "invented_data_NOT_CITYSCAPES"
        image_dir = self.data / "leftImg8bit/val/invented"
        mask_dir = self.data / "gtFine/val/invented"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        for index in range(2):
            name = f"invented_000000_{index:06d}"
            Image.fromarray(np.full((4, 8, 3), 64 + index, np.uint8)).save(
                image_dir / f"{name}_leftImg8bit.png"
            )
            truth = np.full((4, 8), 7, np.uint8)
            truth[:, 4:] = 8
            truth[0, :] = 255
            Image.fromarray(truth).save(mask_dir / f"{name}_gtFine_labelIds.png")
        checkpoint = self.root / "fake_checkpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text(
            json.dumps({"id2label": dict(enumerate(LABELS))})
        )
        (checkpoint / "preprocessor_config.json").write_text("{}")
        bundle = self.root / "fake_bundle"
        bundle.mkdir()
        (bundle / "model.pte").write_bytes(b"TEST DOUBLE NOT AN EXECUTABLE MODEL")
        self.model_sha = checksum(bundle / "model.pte")
        (bundle / "report.json").write_text(
            json.dumps(
                {
                    "model_id": MODEL_ID,
                    "revision": REVISION,
                    "checkpoint_sha256": WEIGHTS_SHA256,
                    "input_shape": [1, 3, 32, 32],
                    "artifacts": {"model.pte": {"sha256": self.model_sha}},
                }
            )
        )
        self.fixtures = self.root / "fixtures"
        self.native = self.root / "fake_native"
        self.native.mkdir()
        args = SimpleNamespace(
            data_root=self.data,
            mask_encoding="label_ids",
            limit=0,
            seed=0,
            bundle=bundle,
            checkpoint=checkpoint,
            output=self.fixtures,
        )
        with patch(
            "export_segformer.load_model", return_value=(FakeModel(), 0)
        ), patch.object(
            torch.Tensor, "cuda", lambda tensor: tensor
        ), contextlib.redirect_stdout(
            io.StringIO()
        ):
            prepare(args)
        self.manifest = json.loads((self.fixtures / "manifest.json").read_text())
        for case in self.manifest["cases"]:
            destination = self.native / case["id"]
            destination.mkdir()
            (destination / "output_0.bin").write_bytes(
                (self.fixtures / case["id"] / "output_0.bin").read_bytes()
            )
            (destination / "loaded-maps.txt").write_text(
                "SYNTHETIC TEST DOUBLE, NOT REAL PROCESS MAPS\n"
            )
        (self.native / "artifact-sha256.txt").write_text(
            self.model_sha + "  /fake_bundle/model.pte\n"
        )
        self.args = SimpleNamespace(
            fixtures=self.fixtures, native=self.native, output=self.root / "scores"
        )

    def run_score(self):
        with contextlib.redirect_stdout(io.StringIO()):
            score(self.args)

    def test_prepare_native_file_score_flow(self):
        self.run_score()
        result = json.loads((self.args.output / "accuracy.json").read_text())
        self.assertEqual(result["num_images"], 2)
        # Original masks are 4x8, not the resized 32x32 model inputs.
        self.assertEqual(result["eager"]["valid_pixels"], 48)
        self.assertEqual(result["eager"]["pixel_accuracy"], 0.5)
        self.assertEqual(result["eager"]["mean_iou"], 0.25)
        self.assertEqual(result["native_minus_eager_miou_percentage_points"], 0)
        self.assertTrue(result["numerical_parity_pass"])

    def test_wrong_model_hash_is_rejected(self):
        (self.native / "artifact-sha256.txt").write_text(
            "wrong  /fake_bundle/model.pte\n"
        )
        with self.assertRaisesRegex(ValueError, "PTE"):
            self.run_score()

    def test_missing_case_is_rejected(self):
        # Rename only this test's own file; no real inference artifacts are touched.
        path = self.native / self.manifest["cases"][0]["id"] / "output_0.bin"
        path.rename(path.with_suffix(".hidden"))
        with self.assertRaisesRegex(ValueError, "Native cases differ"):
            self.run_score()

    def test_changed_label_is_rejected(self):
        truth = self.fixtures / self.manifest["cases"][0]["id"] / "ground_truth.png"
        Image.fromarray(np.zeros((4, 8), np.uint8)).save(truth)
        with self.assertRaisesRegex(ValueError, "Changed or unsafe fixture"):
            self.run_score()

    def test_native_torch_maps_are_rejected(self):
        maps = self.native / self.manifest["cases"][0]["id"] / "loaded-maps.txt"
        maps.write_text("/usr/lib/libtorch.so\n")
        with self.assertRaisesRegex(ValueError, "dependency audit"):
            self.run_score()

    def test_numerical_failure_still_records_metrics(self):
        output = self.native / self.manifest["cases"][0]["id"] / "output_0.bin"
        logits = np.zeros((1, 19, 8, 8), np.float32)
        logits[:, 1] = 1
        logits.tofile(output)
        with self.assertRaisesRegex(SystemExit, "Numerical parity failed"):
            self.run_score()
        result = json.loads((self.args.output / "accuracy.json").read_text())
        self.assertFalse(result["numerical_parity_pass"])


if __name__ == "__main__":
    unittest.main()
