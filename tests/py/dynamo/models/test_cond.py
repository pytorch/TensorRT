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
