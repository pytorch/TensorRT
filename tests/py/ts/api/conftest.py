from pathlib import Path

import pytest

_TS_API_TEST_DIR = Path(__file__).parent


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    """Include TorchScript tests that exercise TensorRT API contracts."""
    for item in items:
        if _TS_API_TEST_DIR in Path(item.path).parents:
            item.add_marker(pytest.mark.trt_api)
