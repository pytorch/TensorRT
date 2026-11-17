import unittest

import pytest
import torch
import torch.nn as nn
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion import DYNAMO_CONVERTERS as CONVERTERS
from torch_tensorrt.dynamo.lowering import (
    get_decompositions,
    post_lowering,
    pre_export_lowering,
)
from torch_tensorrt.dynamo.lowering.passes._aten_lowering_pass import _same_io


def _cond_node(gm: torch.fx.GraphModule) -> torch.fx.Node:
    return next(
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target is torch.ops.higher_order.cond
    )


class _AddSubCond(nn.Module):
    def forward(self, x, predicate):
        return torch.cond(
            predicate,
            lambda value: value + 1,
            lambda value: value - 1,
            (x,),
        )


class _LinearCond(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x, predicate):
        x = torch.relu(self.linear(x))
        return torch.cond(
            predicate,
            lambda value: value + 1,
            lambda value: value - 1,
            (x,),
        )


class _UnsupportedBranchCond(nn.Module):
    def forward(self, x, predicate):
        return torch.cond(
            predicate,
            lambda value: torch.lgamma(value),
            lambda value: value - 1,
            (x,),
        )


class _ComplexCond(nn.Module):
    def forward(self, x, predicate):
        def rotate(value):
            z = torch.view_as_complex(value.reshape(2, 2, 2))
            return torch.view_as_real(z * z).reshape(2, 4)

        return torch.cond(predicate, rotate, lambda value: value - 1, (x,))


class _TransposedComplexCond(nn.Module):
    def forward(self, x, predicate):
        def rotate(value):
            z = torch.view_as_complex(value)
            return torch.view_as_real((z * z).transpose(0, 1))

        result = torch.cond(
            predicate, rotate, lambda value: (value - 1).transpose(0, 1), (x,)
        )
        return torch.as_strided(result, (24,), (1,))


class _SlicedComplexCond(nn.Module):
    def __init__(self, rows):
        super().__init__()
        self.rows = rows

    def forward(self, x, predicate):
        def rotate(value):
            z = torch.view_as_complex(value)
            return torch.view_as_real((z * z)[self.rows])

        result = torch.cond(
            predicate, rotate, lambda value: (value - 1)[self.rows], (x,)
        )
        return torch.as_strided(result, (24,), (1,), 0)


class _DataDependentComplexCond(nn.Module):
    def forward(self, x, n, predicate):
        k = n.item()
        torch._check(k >= 1)
        torch._check(k <= x.shape[0])

        def rotate(value):
            z = torch.view_as_complex(value)
            return torch.view_as_real((z * z).transpose(0, 1))

        return torch.cond(
            predicate,
            rotate,
            lambda value: (value - 1).transpose(0, 1),
            (x[:k].contiguous(),),
        )


class _ComplexOutputCond(nn.Module):
    def forward(self, x, predicate):
        return torch.cond(
            predicate,
            lambda value: torch.view_as_complex(value + 1),
            lambda value: torch.view_as_complex(value - 1),
            (x,),
        )


def _export_lowered(mod: nn.Module, inputs: tuple) -> torch.fx.GraphModule:
    settings = CompilationSettings()
    exported = torch.export.export(mod.eval(), inputs)
    exported = pre_export_lowering(exported, settings)
    exported = exported.run_decompositions(get_decompositions())
    return post_lowering(exported.module(), settings)


@pytest.mark.unit
class TestCondCompilation(TestCase):
    def test_cond_is_supported_when_branches_are_convertible(self):
        x = torch.randn(1, 4)
        pred = torch.tensor(True)
        gm = _export_lowered(_AddSubCond(), (x, pred))
        self.assertTrue(_cond_node(gm) in CONVERTERS)

    def test_cond_keeps_branches_returned_by_lowering(self):
        # The complex lowering returns a new branch module rather than editing it.
        x = torch.randn(2, 4)
        model = _ComplexCond()
        gm = _export_lowered(model, (x, torch.tensor(True)))
        self.assertTrue(_cond_node(gm) in CONVERTERS)
        for pred in (torch.tensor(True), torch.tensor(False)):
            torch.testing.assert_close(gm(x, pred), model(x, pred))

    def test_cond_keeps_branch_when_lowering_changes_output_layout(self):
        pred = torch.tensor(True)
        cases = [
            # The lowered branch is contiguous, so as_strided reads another order.
            (_TransposedComplexCond(), (torch.randn(3, 4, 2), pred)),
            # The lowered branches own smaller storages, one also at another offset.
            (_SlicedComplexCond(slice(1, None)), (torch.randn(4, 3, 2), pred)),
            (_SlicedComplexCond(slice(None, -1)), (torch.randn(4, 3, 2), pred)),
            # Data-dependent strides cannot be proven equal.
            (
                _DataDependentComplexCond(),
                (torch.randn(8, 4, 2), torch.tensor(5), pred),
            ),
        ]
        for model, inputs in cases:
            with self.subTest(type(model).__name__):
                gm = _export_lowered(model, inputs)
                torch.testing.assert_close(gm(*inputs), model(*inputs))

    def test_cond_keeps_branch_when_lowering_changes_output_dtype(self):
        # The lowered branch returns the real [..., 2] layout instead of complex.
        x = torch.randn(2, 2, 2)
        pred = torch.tensor(True)
        model = _ComplexOutputCond()
        gm = _export_lowered(model, (x, pred))
        torch.testing.assert_close(gm(x, pred), model(x, pred))

    def test_cond_branch_check_compares_sizes_and_device(self):
        def identity(value):
            graph = torch.fx.Graph()
            node = graph.placeholder("x")
            node.meta["val"] = value
            graph.output(node)
            return torch.fx.GraphModule(torch.nn.Module(), graph)

        storage = torch.empty(12)
        cases = {
            # Same dtype, strides, storage size and offset.
            "sizes": (storage[:8].view(2, 4), storage.view(3, 4)),
            "device": (torch.empty(2, 4), torch.empty(2, 4, device="meta")),
        }
        for name, (a, b) in cases.items():
            with self.subTest(name):
                self.assertTrue(_same_io(identity(a), identity(a)))
                self.assertFalse(_same_io(identity(a), identity(b)))

    def test_cond_falls_back_when_a_branch_needs_a_pytorch_plugin(self):
        import tensorrt

        if (
            tensorrt.get_plugin_registry().get_creator(
                "ScatterAdd", "1", "torch_tensorrt"
            )
            is None
        ):
            self.skipTest("needs Torch-TensorRT's ScatterAdd plugin")

        # A bool accumulate uses ScatterAdd. A float one would use TensorRT's ScatterElements.
        class AccumulateCond(nn.Module):
            def forward(self, x, index, predicate):
                return torch.cond(
                    predicate,
                    lambda v, i: v.index_put((i,), torch.ones_like(v[:2]), True),
                    lambda v, i: v.logical_not(),
                    (x, index),
                )

        inputs = (
            torch.zeros(8, 4, dtype=torch.bool),
            torch.tensor([1, 1]),
            torch.tensor(True),
        )
        cond = _cond_node(_export_lowered(AccumulateCond(), inputs))
        previous = (CONVERTERS.compilation_settings, CONVERTERS.disallowed_targets)
        try:
            CONVERTERS.set_compilation_settings(CompilationSettings())
            self.assertTrue(cond in CONVERTERS)
            CONVERTERS.set_compilation_settings(
                CompilationSettings(fallback_pytorch_plugin_ops=True)
            )
            self.assertFalse(cond in CONVERTERS)
        finally:
            CONVERTERS.compilation_settings, CONVERTERS.disallowed_targets = previous

    def test_cond_falls_back_when_branch_has_unsupported_op(self):
        x = torch.randn(1, 4).abs() + 0.1
        pred = torch.tensor(True)
        gm = _export_lowered(_UnsupportedBranchCond(), (x, pred))
        self.assertFalse(_cond_node(gm) in CONVERTERS)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_cond_require_full_compilation(self):
        model = _AddSubCond().eval().cuda()
        x = torch.randn(1, 4, device="cuda")

        trt_mod = torch_tensorrt.compile(
            model,
            ir="dynamo",
            inputs=[x, torch.tensor(True, device="cuda")],
            min_block_size=1,
            require_full_compilation=True,
            pass_through_build_failures=True,
            cache_built_engines=False,
            reuse_cached_engines=False,
        )

        for pred in (True, False):
            predicate = torch.tensor(pred, device="cuda")
            eager = model(x, predicate)
            compiled = trt_mod(x, predicate)
            torch.testing.assert_close(compiled, eager, rtol=1e-4, atol=1e-4)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_cond_linear_outside_require_full_compilation(self):
        model = _LinearCond().eval().cuda()
        x = torch.ones(1, 4, device="cuda")

        trt_mod = torch_tensorrt.compile(
            model,
            ir="dynamo",
            inputs=[x, torch.tensor(True, device="cuda")],
            min_block_size=1,
            require_full_compilation=True,
            pass_through_build_failures=True,
            cache_built_engines=False,
            reuse_cached_engines=False,
        )

        for pred in (True, False):
            predicate = torch.tensor(pred, device="cuda")
            eager = model(x, predicate)
            compiled = trt_mod(x, predicate)
            torch.testing.assert_close(compiled, eager, rtol=1e-4, atol=1e-4)
