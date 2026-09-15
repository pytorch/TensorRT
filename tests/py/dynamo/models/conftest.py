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


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "torchao: TorchAO quantization compile tests; collected by the dynamo-torchao full-lane suite",
    )


def pytest_collection_modifyitems(config, items):
    """Mark TorchAO model tests so the dynamo-torchao suite can own them."""
    marker = pytest.mark.torchao
    for item in items:
        path = str(getattr(item, "path", item.fspath))
        if "test_torchao" in path:
            item.add_marker(marker)
