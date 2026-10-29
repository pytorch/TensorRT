# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small dependency-free failure-path tests for the standalone box decoder."""

import argparse
import struct
import subprocess
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("decoder", type=Path)
args = parser.parse_args()
binary = args.decoder.resolve()
with tempfile.TemporaryDirectory(prefix="bev-decoder-negative-") as temporary:
    root = Path(temporary)
    for name, expected in (
        ("size", "Bad size"),
        ("nan", "Nonfinite raw prediction"),
        ("existing", "Output exists"),
    ):
        case = root / name
        case.mkdir()
        logits = [0.0] * 9000
        if name == "nan":
            logits[0] = float("nan")
        (case / "output_0.bin").write_bytes(struct.pack("<9000f", *logits))
        (case / "output_1.bin").write_bytes(b"" if name == "size" else bytes(36000))
        output = case / "decoded.tsv"
        if name == "existing":
            output.write_text("preserve me\n")
        result = subprocess.run(
            [str(binary), str(case), str(output)], capture_output=True, text=True
        )
        assert result.returncode != 0 and expected in result.stderr, result
        if name == "existing":
            assert output.read_text() == "preserve me\n"
        print("PASS: rejects", name, "without silently accepting bad evidence")
