import operator
import unittest
from typing import Tuple

import torch
from torch.testing._internal.common_utils import run_tests

from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_CONVERTERS,
    dynamo_tensorrt_converter,
)
from torch_tensorrt.dynamo.conversion.plugins._auto_functionalized_converter import (
    _ALIAS_MAP_CACHE,
    _WRAPPER_TARGETS,
    _is_matching_writeback,
    _output_alias_map,
    _reconstruct_op_args,
    wrapper_wraps_plugin_op,
)
from torch_tensorrt.dynamo.conversion.plugins._generate_plugin import (
    _validate_mutating_schema,
)


def _dummy_converter(ctx, target, args, kwargs, name):
    return args[0]


@torch.library.custom_op("torchtrt_inplace_validation::allowed", mutates_args=("x",))
def _allowed(x: torch.Tensor) -> torch.Tensor:
    x.add_(1)
    return x.clone()


@_allowed.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return x


_ALLOWED_OP = torch.ops.torchtrt_inplace_validation.allowed.default
dynamo_tensorrt_converter(
    _ALLOWED_OP,
    requires_aliased_plugin_io=True,
)(_dummy_converter)


@torch.library.custom_op("torchtrt_inplace_validation::rejected", mutates_args=("x",))
def _rejected(x: torch.Tensor) -> torch.Tensor:
    x.add_(1)
    return x.clone()


@_rejected.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return x


_REJECTED_OP = torch.ops.torchtrt_inplace_validation.rejected.default
dynamo_tensorrt_converter(
    _REJECTED_OP,
    capability_validator=lambda node, settings: False,
    requires_aliased_plugin_io=True,
)(_dummy_converter)


@torch.library.custom_op(
    "torchtrt_inplace_validation::missing_alias", mutates_args=("x",)
)
def _missing_alias(x: torch.Tensor) -> torch.Tensor:
    x.add_(1)
    return x.clone()


@_missing_alias.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


_MISSING_ALIAS_OP = torch.ops.torchtrt_inplace_validation.missing_alias.default
dynamo_tensorrt_converter(
    _MISSING_ALIAS_OP,
    requires_aliased_plugin_io=True,
)(_dummy_converter)


@torch.library.custom_op("torchtrt_inplace_validation::cuda_fake", mutates_args=("x",))
def _cuda_fake(x: torch.Tensor) -> torch.Tensor:
    x.add_(1)
    return x.clone()


@_cuda_fake.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    return x


_CUDA_FAKE_OP = torch.ops.torchtrt_inplace_validation.cuda_fake.default


@torch.library.custom_op("torchtrt_inplace_validation::multi", mutates_args=("x",))
def _multi(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x.add_(y)
    return x.clone(), x * 2


@_multi.register_fake
def _(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return x, torch.empty_like(x)


_MULTI_OP = torch.ops.torchtrt_inplace_validation.multi.default
dynamo_tensorrt_converter(
    _MULTI_OP,
    requires_aliased_plugin_io=True,
)(_dummy_converter)


_SCHEMA_LIBRARY = torch.library.Library("torchtrt_inplace_schema_validation", "DEF")
_SCHEMA_LIBRARY.define("returns_nothing(Tensor(a!) x) -> ()")
_SCHEMA_LIBRARY.define("returns_none(Tensor(a!) x) -> None")

_RETURNS_NOTHING_OP = (
    torch.ops.torchtrt_inplace_schema_validation.returns_nothing.default
)
_RETURNS_NONE_OP = torch.ops.torchtrt_inplace_schema_validation.returns_none.default


def _functionalized_module(model: torch.nn.Module, *inputs: torch.Tensor):
    exported = torch.export.export(model, inputs, strict=False)
    return exported.run_decompositions({}).module()


def _wrapper(graph_module: torch.fx.GraphModule) -> torch.fx.Node:
    return next(
        node
        for node in graph_module.graph.nodes
        if node.target is torch.ops.higher_order.auto_functionalized_v2
    )


class TestAutoFunctionalizedPluginValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._previous_settings = DYNAMO_CONVERTERS.compilation_settings
        cls._previous_disallowed = DYNAMO_CONVERTERS.disallowed_targets

    @classmethod
    def tearDownClass(cls):
        DYNAMO_CONVERTERS.compilation_settings = cls._previous_settings
        DYNAMO_CONVERTERS.disallowed_targets = cls._previous_disallowed

    def setUp(self):
        DYNAMO_CONVERTERS.set_compilation_settings(CompilationSettings())

    def test_delegates_to_inner_capability_validator(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return _REJECTED_OP(x)

        wrapper = _wrapper(_functionalized_module(Model(), torch.randn(4, 4)))
        self.assertFalse(wrapper_wraps_plugin_op(wrapper, CompilationSettings()))

    def test_delegates_dynamic_shape_support_to_inner_converter(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return _ALLOWED_OP(x)

        batch = torch.export.Dim("batch", min=1, max=8)
        exported = torch.export.export(
            Model(),
            (torch.randn(4, 4),),
            dynamic_shapes={"x": {0: batch}},
            strict=False,
        )
        wrapper = _wrapper(exported.run_decompositions({}).module())
        self.assertFalse(wrapper_wraps_plugin_op(wrapper, CompilationSettings()))

    def test_rejects_mutated_views(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return _ALLOWED_OP(x[:, :2])

        wrapper = _wrapper(_functionalized_module(Model(), torch.randn(4, 4)))
        self.assertIn("_x_slice_dim", wrapper.kwargs)
        self.assertFalse(wrapper_wraps_plugin_op(wrapper, CompilationSettings()))
        with self.assertRaisesRegex(RuntimeError, "can mutate only a base tensor"):
            _reconstruct_op_args(_ALLOWED_OP, dict(wrapper.kwargs))

    def test_missing_fake_alias_is_a_hard_error(self):
        _ALIAS_MAP_CACHE.pop(_MISSING_ALIAS_OP, None)
        with self.assertRaisesRegex(
            RuntimeError, "did not return the same FakeTensor object"
        ):
            _output_alias_map(_MISSING_ALIAS_OP, [torch.randn(4, 4)])

    def test_alias_detection_uses_cuda_fake_tensors(self):
        _ALIAS_MAP_CACHE.pop(_CUDA_FAKE_OP, None)
        self.assertEqual(
            _output_alias_map(_CUDA_FAKE_OP, [torch.randn(4, 4)]),
            {0: 0},
        )

    def test_writeback_must_target_its_matching_base_slot(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        wrapper = graph.call_function(
            torch.ops.higher_order.auto_functionalized_v2,
            args=(_ALLOWED_OP,),
            kwargs={"_x_base_index": 0, "_all_bases": [x]},
        )
        return_value = graph.call_function(operator.getitem, args=(wrapper, 0))
        updated_base = graph.call_function(operator.getitem, args=(wrapper, 1))

        wrong_slot = graph.call_function(
            torch.ops.aten.copy_.default, args=(x, return_value)
        )
        wrong_base = graph.call_function(
            torch.ops.aten.copy_.default, args=(y, updated_base)
        )
        matching = graph.call_function(
            torch.ops.aten.copy_.default, args=(x, updated_base)
        )

        self.assertFalse(_is_matching_writeback(wrong_slot))
        self.assertFalse(_is_matching_writeback(wrong_base))
        self.assertTrue(_is_matching_writeback(matching))

    def test_rejects_consumed_alias_from_multi_output_plugin(self):
        class UnsafeModel(torch.nn.Module):
            def forward(self, x, y):
                aliased, _ = _MULTI_OP(x, y)
                return aliased * 2

        class SafeModel(torch.nn.Module):
            def forward(self, x, y):
                _, fresh = _MULTI_OP(x, y)
                return fresh

        inputs = (torch.randn(4, 4), torch.randn(4, 4))
        unsafe = _wrapper(_functionalized_module(UnsafeModel(), *inputs))
        safe = _wrapper(_functionalized_module(SafeModel(), *inputs))
        self.assertFalse(wrapper_wraps_plugin_op(unsafe, CompilationSettings()))
        self.assertTrue(wrapper_wraps_plugin_op(safe, CompilationSettings()))

    def test_v1_auto_functionalized_is_not_registered(self):
        self.assertNotIn(
            torch.ops.higher_order.auto_functionalized,
            _WRAPPER_TARGETS,
        )

    def test_none_returning_mutations_are_rejected_clearly(self):
        for op in (_RETURNS_NOTHING_OP, _RETURNS_NONE_OP):
            with self.subTest(op=op), self.assertRaisesRegex(
                ValueError, "None or non-Tensor-only returns are not supported"
            ):
                _validate_mutating_schema(op._schema, str(op))


if __name__ == "__main__":
    run_tests()
