# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# type: ignore

import pytest


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
