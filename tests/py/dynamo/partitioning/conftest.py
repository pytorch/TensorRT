import copy

import pytest
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_ATEN_CONVERTERS,
    DYNAMO_CONVERTERS,
)
from torch_tensorrt.dynamo.partitioning._atomic_subgraphs import trace_atomic_graph


@pytest.fixture(autouse=True)
def reset_torch_tensorrt_state():
    """
    Ensure test isolation by restoring converter registry state and clearing caches.
    This prevents earlier tests from mutating global state (e.g., disallowed targets)
    which can cause different partitioning outcomes when running multiple tests.
    """
    # Snapshot current global state
    original_registry = {k: list(v) for k, v in DYNAMO_ATEN_CONVERTERS.items()}
    original_disallowed = set(getattr(DYNAMO_CONVERTERS, "disallowed_targets", set()))
    original_settings = getattr(DYNAMO_CONVERTERS, "compilation_settings", None)

    # Clear caches before running each test
    try:
        trace_atomic_graph.cache_clear()
    except Exception:
        pass

    try:
        yield
    finally:
        # Restore converter registry
        DYNAMO_ATEN_CONVERTERS.clear()
        DYNAMO_ATEN_CONVERTERS.update(
            {k: list(v) for k, v in original_registry.items()}
        )

        # Restore disallowed targets and compilation settings
        try:
            DYNAMO_CONVERTERS.set_disallowed_targets(original_disallowed)
        except Exception:
            pass
        if original_settings is not None:
            try:
                DYNAMO_CONVERTERS.set_compilation_settings(original_settings)
            except Exception:
                pass

        # Clear caches again to avoid stale state carrying forward
        try:
            trace_atomic_graph.cache_clear()
        except Exception:
            pass
