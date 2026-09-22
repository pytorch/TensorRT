# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# type: ignore

from pathlib import Path

import pytest

_RUNTIME_TEST_DIR = Path(__file__).parent


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    """Include runtime tests that exercise TensorRT runtime APIs in trt-api."""
    for item in items:
        if _RUNTIME_TEST_DIR in Path(item.path).parents:
            item.add_marker(pytest.mark.trt_api)


def pytest_addoption(parser):
    try:
        parser.addoption(
            "--ir",
            metavar="Internal Representation",
            nargs=1,
            type=str,
            required=False,
            help="IR to compile with",
            choices=["dynamo", "torch_compile"],
        )
    except ValueError:
        pass  # --ir already registered by another conftest (e.g. models/conftest.py)


@pytest.fixture
def ir(request):
    ir_opt = request.config.getoption("--ir")
    return ir_opt[0] if ir_opt else "dynamo"
