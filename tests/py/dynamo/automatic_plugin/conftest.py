from pathlib import Path

import pytest

_PLUGIN_TEST_DIR = Path(__file__).parent


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    """Include tests of TensorRT's plugin APIs in trt-api."""
    for item in items:
        if _PLUGIN_TEST_DIR in Path(item.path).parents:
            item.add_marker(pytest.mark.trt_api)
