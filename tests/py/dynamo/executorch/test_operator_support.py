from types import SimpleNamespace

import pytest
from torch_tensorrt.executorch.operator_support import TensorRTOperatorSupport


class _SchemaTarget:
    def __init__(self, name):
        self._schema = SimpleNamespace(name=name)


@pytest.mark.unit
def test_operator_support_accepts_execute_engine_variants():
    support = TensorRTOperatorSupport()

    assert support.is_node_supported(
        {},
        SimpleNamespace(
            op="call_function",
            target=_SchemaTarget("tensorrt::execute_engine"),
        ),
    )
    assert support.is_node_supported(
        {},
        SimpleNamespace(
            op="call_function",
            target=_SchemaTarget("tensorrt::no_op_placeholder_for_execute_engine"),
        ),
    )
    assert not support.is_node_supported(
        {},
        SimpleNamespace(
            op="call_function",
            target=_SchemaTarget("aten::add"),
        ),
    )
    assert not support.is_node_supported(
        {},
        SimpleNamespace(
            op="placeholder",
            target=_SchemaTarget("tensorrt::execute_engine"),
        ),
    )
