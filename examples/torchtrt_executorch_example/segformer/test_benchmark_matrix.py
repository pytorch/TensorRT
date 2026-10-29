# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Synthetic report-parser controls; these are not inference measurements."""

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from summarize_matrix import checked_python_run


class MatrixTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)
        self.report = {
            "mode": "eager",
            "model_family": "segformer",
            "warmup": 20,
            "iterations": 100,
            "comparisons": [
                dict(
                    phase=p, output=0, elements=1245184, mismatches=0, max_abs_error=0.0
                )
                for p in ("first", "after_warmup", "after_measurement")
            ],
            "memory": [
                dict(
                    phase="after_measurement",
                    errors=[],
                    rss_kib=1024,
                    pss_kib=512,
                    process_hwm_kib=2048,
                )
            ],
            "median_ms": 1.0,
            "versions": {},
            "fixture_sha256": {},
            "artifact_sha256": {},
        }
        (self.path / "PASS").write_text("TEST DOUBLE")
        (self.path / "loaded-maps.txt").write_text("TEST DOUBLE /libtorch.so")
        with (self.path / "latency.tsv").open("w") as file:
            writer = csv.writer(file, delimiter="\t")
            writer.writerow(["phase", "iteration", "wall_ms"])
            for phase, count in (("first", 1), ("warmup", 20), ("measured", 100)):
                writer.writerows((phase, i, 1.0) for i in range(count))

    def run_summary(self):
        (self.path / "report.json").write_text(json.dumps(self.report))
        return checked_python_run(self.path, "eager")

    def test_valid(self):
        result = self.run_summary()
        self.assertEqual(result["rss_mib"], 1.0)
        self.assertEqual(result["p95_ms"], 1.0)

    def test_protocol_and_correctness_failures(self):
        original = copy.deepcopy(self.report)
        for key, value in (
            ("model_family", "bevformer"),
            ("warmup", 0),
            ("iterations", 99),
            ("median_ms", 99.0),
            ("comparisons", original["comparisons"][:-1]),
        ):
            with self.subTest(key=key):
                self.report = copy.deepcopy(original)
                self.report[key] = value
                with self.assertRaises(ValueError):
                    self.run_summary()
        self.report = copy.deepcopy(original)
        self.report["comparisons"][0]["mismatches"] = 1
        with self.assertRaises(ValueError):
            self.run_summary()

    def test_missing_pss_is_not_zero(self):
        self.report["memory"][0]["pss_kib"] = None
        with self.assertRaises(ValueError):
            self.run_summary()

    def test_missing_torch_control_is_rejected(self):
        (self.path / "loaded-maps.txt").write_text("test double without runtime")
        with self.assertRaises(ValueError):
            self.run_summary()


if __name__ == "__main__":
    unittest.main()
