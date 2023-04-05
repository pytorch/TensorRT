# Owner(s): ["oncall: gpu_enablement"]

import logging
from copy import deepcopy

import torch
import torch.fx as fx
import torch.nn as nn

from torch.testing._internal.common_utils import run_tests, TestCase
from torch_tensorrt.dynamo.passes.lower_basic_pass import fix_reshape_batch_dim
from torch_tensorrt.fx.tracer.acc_tracer import acc_tracer

_LOGGER = logging.getLogger(__name__)


class TestFixReshapeBatchDim(TestCase):
    def test_fix_reshape_batch_dim(self):
        class Repro(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return y.view(x.size(0), -1, 3)

        mod = Repro()
        modt = fx.symbolic_trace(mod)
        inp = [
            torch.rand([10, 60]),
            torch.rand([10, 60]),
        ]
        mod(*inp)
        mod_acc_traced = acc_tracer.trace(modt, inp)
        mod_fixed = fix_reshape_batch_dim(deepcopy(mod_acc_traced))

        expected_graph = r"""
graph():
    %x : [#users=0] = placeholder[target=x]
    %y : [#users=2] = placeholder[target=y]
    %size : [#users=1] = call_function[target=torch_tensorrt.fx.tracer.acc_tracer.acc_ops.size](args = (), kwargs = {input: %y})
    %getitem_1 : [#users=1] = call_function[target=torch_tensorrt.fx.tracer.acc_tracer.acc_ops.getitem](args = (), kwargs = {idx: 0, input: %size})
    %reshape : [#users=1] = call_function[target=torch_tensorrt.fx.tracer.acc_tracer.acc_ops.reshape](args = (), kwargs = {input: %y, acc_out_ty: ((%getitem_1, -1, 3), None, None, None, None, None, None)})
    return reshape
"""
        assert (
            str(mod_fixed.graph).strip() == expected_graph.strip()
        ), f"Unexpected fixed graph. \nActual: {str(mod_fixed.graph)} \nExpected: {expected_graph}"


if __name__ == "__main__":
    run_tests()
