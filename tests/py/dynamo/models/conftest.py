# type: ignore

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--ir",
        metavar="Internal Representation",
        nargs=1,
        type=str,
        required=False,
        help="IR to compile with",
        choices=["dynamo", "torch_compile"],
    )


@pytest.fixture
def ir(request):
    ir_opt = request.config.getoption("--ir")
    return ir_opt[0] if ir_opt else "dynamo"
