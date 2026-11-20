# SPDX-License-Identifier: BSD-3-Clause
"""The summary must reject corrupt artifacts and incomplete benchmark evidence."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from run import native_files, result, verify


class HandoffChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_rows(self, name, rows):
        with (self.root / name).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0], delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    def evidence(self):
        (self.root / "PASS").touch()
        self.write_rows(
            "latency.tsv",
            [
                dict(phase=p, iteration=i, wall_ms=i + 1)
                for p, n in (("first", 1), ("warmup", 20), ("measured", 100))
                for i in range(n)
            ],
        )
        self.write_rows(
            "comparisons.tsv",
            [
                dict(phase=p, output=0, elements=10, mismatches=0)
                for p in ("first", "after_warmup", "after_measurement")
            ],
        )
        self.write_rows(
            "memory.tsv",
            [
                dict(
                    phase=p,
                    rss_kib=1024,
                    pss_kib=512,
                    process_hwm_kib=2048,
                    cuda_free_bytes=free,
                )
                for p, free in (
                    ("cuda_initialized", 2097152),
                    ("method_loaded", 1048576),
                    ("after_measurement", 1048576),
                )
            ],
        )

    def test_summary_uses_process_median_and_distinct_memory_metrics(self):
        self.evidence()
        r = result(self.root, "segformer")
        self.assertEqual(r["latency_ms"], 50.5)
        self.assertEqual(r["loading_peak_rss_mib"], 2)
        self.assertEqual(r["retained_rss_mib"], 1)
        self.assertEqual(r["retained_pss_mib"], 0.5)
        self.assertEqual(r["cuda_free_decrease_mib"], 1)

    def test_partial_timing_is_rejected_even_with_pass_marker(self):
        self.evidence()
        p = self.root / "latency.tsv"
        p.write_text("\n".join(p.read_text().splitlines()[:-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "Incomplete measured"):
            result(self.root, "segformer")

    def test_missing_output_checks_are_rejected(self):
        self.evidence()
        with self.assertRaisesRegex(ValueError, "Missing output checks"):
            result(self.root, "bevformer")

    def test_failed_comparison_is_rejected_even_with_pass_marker(self):
        self.evidence()
        self.write_rows(
            "comparisons.tsv",
            [
                dict(phase=p, output=0, elements=10, mismatches=1)
                for p in ("first", "after_warmup", "after_measurement")
            ],
        )
        with self.assertRaisesRegex(ValueError, "Output mismatch"):
            result(self.root, "segformer")

    def test_artifact_integrity_and_path_boundary(self):
        data = b"artifact"
        (self.root / "model.pte").write_bytes(data)
        entry = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(dict(format=1, files={"model.pte": entry})))
        verify(self.root)
        (self.root / "model.pte").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            verify(self.root)
        manifest.write_text(json.dumps(dict(format=1, files={"../outside": entry})))
        with self.assertRaisesRegex(ValueError, "Invalid artifact path"):
            verify(self.root)

    def test_rebuilt_runtime_requires_libraries_and_records_exact_bytes(self):
        with self.assertRaisesRegex(ValueError, "Missing rebuilt"):
            native_files(self.root)
        names = (
            "bin/benchmark_et",
            "bin/benchmark_trt",
            "lib/libaoti_cuda_shims.so",
            "lib/libextension_cuda.so",
        )
        for name in names:
            path = self.root / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(name.encode())
        measured = native_files(self.root)
        self.assertEqual(
            measured[names[0]], hashlib.sha256(names[0].encode()).hexdigest()
        )
        (self.root / names[-1]).unlink()
        with self.assertRaisesRegex(ValueError, "Missing rebuilt"):
            native_files(self.root)


if __name__ == "__main__":
    unittest.main()
