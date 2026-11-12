# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""The Auto-Tuner must ship in the wheel: python -m torch_tensorrt.autotune needs it."""

import ast
from pathlib import Path

from setuptools import Distribution
from setuptools.command.build_py import build_py


def test_the_wheel_includes_the_autotune_package():
    root = Path(__file__).resolve().parents[4]
    setup = root / "setup.py"
    declarations = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(setup.read_text(encoding="utf-8")).body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"dynamo_packages", "dynamo_package_dir"}
    }
    distribution = Distribution(
        {
            "packages": declarations["dynamo_packages"],
            "package_dir": {
                package: str(root / directory)
                for package, directory in declarations["dynamo_package_dir"].items()
            },
        }
    )
    distribution.script_name = str(setup)
    command = build_py(distribution)
    command.ensure_finalized()
    selected = {Path(path).resolve() for path in command.get_source_files()}
    required = set((root / "py/torch_tensorrt/autotune").glob("*.py"))
    assert required, "py/torch_tensorrt/autotune has no modules"
    assert required <= selected, f"the wheel omits {sorted(required - selected)}"
