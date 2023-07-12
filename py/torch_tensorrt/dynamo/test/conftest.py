import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--ir",
        metavar="Internal Representation",
        nargs=1,
        type=str,
        required=True,
        help="IR to compile with",
        choices=["dynamo", "torch_compile"],
    )


@pytest.fixture
def ir(request):
    return request.config.getoption("--ir")[0]
