# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Region modules must ship in the Python-only wheel as well as native builds."""

import ast
from pathlib import Path

from setuptools import Distribution
from setuptools.command.build_py import build_py


def test_python_only_wheel_includes_region_modules():
    root = Path(__file__).resolve().parents[4]
    setup = root / "setup.py"
    declarations = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(setup.read_text(encoding="utf-8")).body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id
        in {"dynamo_packages", "dynamo_package_dir", "package_data"}
    }
    distribution = Distribution(
        {
            "packages": declarations["dynamo_packages"],
            "package_dir": {
                package: str(root / directory)
                for package, directory in declarations["dynamo_package_dir"].items()
            },
            "package_data": declarations["package_data"],
        }
    )
    distribution.script_name = str(setup)
    command = build_py(distribution)
    command.ensure_finalized()
    selected = {Path(path).resolve() for path in command.get_source_files()}
    selected.update(
        Path(source, name).resolve()
        for _, source, _, names in command.data_files
        for name in names
    )
    package = root / "py/torch_tensorrt"
    directories = (
        "region",
        "dynamo/region_compilation",
        "dynamo/region_compilation/_generation",
    )
    required = {
        path for directory in directories for path in (package / directory).glob("*.py")
    }
    # Kernel generation briefs the agent with the packaged kernel-author skill.
    required.update(
        (package / "dynamo/region_compilation/_generation/skills").rglob("*.md")
    )
    assert required <= selected, f"Python-only wheel omits {required - selected}"
