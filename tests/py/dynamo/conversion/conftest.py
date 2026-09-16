from pathlib import Path

import pytest

_CONVERTER_TEST_DIR = Path(__file__).parent


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    """Include every converter test in the TensorRT API contract suite."""
    for item in items:
        if _CONVERTER_TEST_DIR in Path(item.path).parents:
            item.add_marker(pytest.mark.trt_api)
