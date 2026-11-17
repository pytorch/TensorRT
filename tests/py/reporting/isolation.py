"""Load stdlib modules without executing the native-heavy package initializer."""

import importlib.util
import sys
import types
from pathlib import Path


def _load(child):
    root = Path(__file__).resolve().parents[3] / "py" / "torch_tensorrt"
    name = "_torch_tensorrt_contract_tests"
    module_name = name + "." + child
    if module_name in sys.modules:
        return sys.modules[module_name]
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(root)]
        sys.modules[name] = package
    spec = importlib.util.spec_from_file_location(
        module_name,
        root / child / "__init__.py",
        submodule_search_locations=[str(root / child)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_reporting():
    return _load("reporting")
