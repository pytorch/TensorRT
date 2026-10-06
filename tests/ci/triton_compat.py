# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Run kernel regressions with a selected Triton wheel without replacing the installed one.

Usage: python -m tests.ci.triton_compat [3.5.0]
Requires a CUDA/TensorRT test environment. Only the child process sees the wheel.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    version = sys.argv[1] if len(sys.argv) > 1 else "3.5.0"
    with tempfile.TemporaryDirectory(prefix="triton-compat-") as directory:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--only-binary=:all:",
                "--target",
                directory,
                f"triton=={version}",
            ],
            check=True,
        )
        env = dict(
            os.environ,
            PYTHONPATH=os.pathsep.join(
                filter(None, [directory, os.environ.get("PYTHONPATH")])
            ),
        )
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, triton, pytest; "
                "assert triton.__version__ == sys.argv[1], triton.__version__; "
                "raise SystemExit(pytest.main(sys.argv[2:]))",
                version,
                "-q",
                "-o",
                "addopts=",
                "tests/py/kernels/test_triton_op.py",
            ],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
