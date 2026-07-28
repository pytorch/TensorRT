# type: ignore

import gc

import pytest
import torch


def _release_trt_engines():
    """Force-collect any lingering TRT engine objects before the session ends.

    TRT ICudaEngine / IExecutionContext objects hold GPU memory.  Python's GC
    does not guarantee ordering of destructor calls, so engines may outlive
    their parent modules.  Calling gc.collect() twice (generation 0→1→2)
    ensures that all reference cycles are broken and __del__ is called before
    CUDA shuts down.
    """
    gc.collect()
    gc.collect()


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
        pass  # --ir already registered by another conftest


@pytest.fixture
def ir(request):
    ir_opt = request.config.getoption("--ir")
    return ir_opt[0] if ir_opt else "dynamo"


def pytest_sessionstart(session):
    # Disable Python's cyclic garbage collector for the duration of the test
    # session.  TRT objects hold raw C++ pointers; if two Python objects that
    # reference each other (a cycle) are collected together, the order in which
    # __del__ is called is undefined, and the TRT engine destructor can run
    # after CUDA has already been torn down, causing a segfault.
    # With GC disabled we rely on reference counting alone; objects are
    # destroyed deterministically when the last reference drops.
    gc.disable()


def pytest_sessionfinish(session, exitstatus):
    _release_trt_engines()
    gc.enable()


@pytest.fixture(autouse=True)
def gpu_cleanup():
    yield
    # After every test: release lingering TRT objects and flush the CUDA cache
    # so that the next test starts with a clean GPU memory state.
    _release_trt_engines()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
