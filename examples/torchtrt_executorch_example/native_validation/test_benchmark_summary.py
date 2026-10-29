# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import tempfile
import unittest
from pathlib import Path
from summarize_benchmark import percentile, summarize_run


class SummaryTests(unittest.TestCase):
    def test_percentiles(self):
        self.assertEqual(percentile([1, 2, 3, 4, 5], 0.5), 3)
        self.assertAlmostEqual(percentile([1, 2, 3, 4, 5], 0.95), 4.8)
        self.assertEqual(percentile([2], 0.95), 2)

    def make_run(self, root):
        (root / "PASS").write_text("PASS\n")
        (root / "metadata.tsv").write_text("key\tvalue\niterations\t2\nwarmup\t1\n")
        (root / "comparisons.tsv").write_text(
            "phase\toutput\telements\tmax_abs_error\tmismatches\n"
            "first\t0\t3\t0.00001\t0\n"
            "after_warmup\t0\t3\t0.00001\t0\n"
            "after_measurement\t0\t3\t0.00001\t0\n"
        )
        (root / "latency.tsv").write_text(
            "phase\titeration\twall_ms\nfirst\t0\t10\nwarmup\t0\t4\nmeasured\t0\t2\nmeasured\t1\t3\n"
        )
        (root / "loaded-maps.txt").write_text("/lib/libnvinfer.so\n")
        (root / "memory.tsv").write_text("phase\trss_kib\nfirst\t1024\n")
        (root / "phases.tsv").write_text("phase\twall_ms\nmodule_load\t3\n")

    def test_valid_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_run(root)
            result = summarize_run(root)
            self.assertEqual(result["median_ms"], 2.5)
            self.assertEqual(result["compared_elements_per_gate"], 3)

    def test_reject_invalid_evidence(self):
        changes = {
            "PASS": None,
            "comparisons.tsv": "phase\toutput\telements\tmax_abs_error\tmismatches\nfirst\t0\t3\t1\t1\n",
            "latency.tsv": "phase\titeration\twall_ms\nfirst\t0\tnan\n",
            "loaded-maps.txt": "/some/path/libtorch.so\n",
        }
        for name, contents in changes.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.make_run(root)
                if contents is None:
                    (root / name).unlink()
                else:
                    (root / name).write_text(contents)
                with self.assertRaises(ValueError):
                    summarize_run(root)


if __name__ == "__main__":
    unittest.main()
