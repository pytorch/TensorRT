# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import process_memory


class ProcessMemoryTests(unittest.TestCase):
    def test_parse_exact_field_and_zero(self):
        text = "VmRSSExtra: 999 kB\nVmRSS:\t2048 kB\nPss: 0 kB\n"
        self.assertEqual(process_memory.parse_kib(text, "VmRSS"), 2048)
        self.assertEqual(process_memory.parse_kib(text, "Pss"), 0)

    def test_missing_or_invalid_measurement_is_not_zero(self):
        for text in ("Other: 0 kB", "Pss: 100 MB", "Pss: -1 kB", "Pss: bad kB"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                process_memory.parse_kib(text, "Pss")

    def make_proc(self, directory):
        process = Path(directory) / "123"
        process.mkdir()
        (process / "status").write_text(
            "Name: test\nVmRSS: 204800 kB\nVmHWM: 256000 kB\n"
        )
        (process / "smaps_rollup").write_text("Rss: 204800 kB\nPss: 153600 kB\n")
        return process

    def test_snapshot_units_and_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            self.make_proc(directory)
            result = process_memory.snapshot("loaded", pid=123, proc_root=directory)
        self.assertEqual(result["pid"], 123)
        self.assertEqual(result["phase"], "loaded")
        self.assertEqual(result["rss_kib"] / 1024, 200)
        self.assertEqual(result["pss_kib"] / 1024, 150)
        self.assertEqual(result["process_hwm_kib"] / 1024, 250)
        self.assertEqual(result["smaps_rss_kib"], result["rss_kib"])
        self.assertEqual(result["errors"], [])

    def test_unavailable_pss_preserves_rss(self):
        with tempfile.TemporaryDirectory() as directory:
            process = self.make_proc(directory)
            (process / "smaps_rollup").unlink()
            result = process_memory.snapshot("loaded", pid=123, proc_root=directory)
        self.assertEqual(result["rss_kib"], 204800)
        self.assertIsNone(result["pss_kib"])
        self.assertTrue(result["errors"])

    def test_missing_process_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            result = process_memory.snapshot("exited", pid=123, proc_root=directory)
        for field in ("rss_kib", "pss_kib", "smaps_rss_kib", "process_hwm_kib"):
            self.assertIsNone(result[field])
        self.assertEqual(len(result["errors"]), 2)

    def test_invalid_arguments(self):
        for pid in (0, -1, True, "123"):
            with self.subTest(pid=pid), self.assertRaises(ValueError):
                process_memory.snapshot("test", pid=pid)
        with self.assertRaises(ValueError):
            process_memory.snapshot(" ")

    def test_append_keeps_previous_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            self.make_proc(directory)
            first = process_memory.snapshot("before", pid=123, proc_root=directory)
            second = {**first, "phase": "after"}
            output = Path(directory) / "memory.jsonl"
            with patch.object(process_memory, "snapshot", side_effect=[first, second]):
                with contextlib.redirect_stdout(io.StringIO()) as console:
                    process_memory.log_memory("before", output)
                    process_memory.log_memory("after", output)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual([row["phase"] for row in rows], ["before", "after"])
        self.assertIn("PSS=150.00 MiB", console.getvalue())

    def test_cli_returns_failure_for_unavailable_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            result = process_memory.snapshot("snapshot", pid=123, proc_root=directory)
        with patch.object(process_memory, "log_memory", return_value=result):
            with patch("sys.argv", ["process_memory.py", "--pid", "123"]):
                self.assertEqual(process_memory.main(), 1)


if __name__ == "__main__":
    unittest.main()
