# Owner(s): ["oncall: gpu_enablement"]

import logging
from copy import deepcopy
from packaging import version

import torch
import torch.fx as fx
import torch.nn as nn

from torch.testing._internal.common_utils import run_tests, TestCase
from torch_tensorrt.fx.passes.lower_basic_pass import fix_reshape_batch_dim
from torch_tensorrt.fx.tracer.acc_tracer import acc_tracer
from torch_tensorrt._utils import sanitized_torch_version

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
    %x : [num_users=0] = placeholder[target=x]
    %y : [num_users=2] = placeholder[target=y]
    %size : [num_users=1] = call_function[target=torch_tensorrt.fx.tracer.acc_tracer.acc_ops.size](args = (), kwargs = {input: %y})
    %getitem_1 : [num_users=1] = call_function[target=torch_tensorrt.fx.tracer.acc_tracer.acc_ops.getitem](args = (), kwargs = {idx: 0, input: %size})
    %reshape : [num_users=1] = call_function[target=torch_tensorrt.fx.tracer.acc_tracer.acc_ops.reshape](args = (), kwargs = {input: %y, acc_out_ty: ((%getitem_1, -1, 3), None, None, None, None, None, None)})
    return reshape
"""
        if version.parse(sanitized_torch_version()) < version.parse(
            "2.1.0.dev20230620"
        ):
            expected_graph = expected_graph.replace("num_users", "#users")

        assert (
            str(mod_fixed.graph).strip() == expected_graph.strip()
        ), f"Unexpected fixed graph. \nActual: {str(mod_fixed.graph)} \nExpected: {expected_graph}"


if __name__ == "__main__":
    run_tests()
