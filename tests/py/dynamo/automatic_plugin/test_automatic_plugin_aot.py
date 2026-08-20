import platform
import unittest
from typing import List, Tuple

import torch
import torch.nn as nn
import torch_tensorrt
from torch.testing._internal.common_utils import run_tests

from ..conversion.harness import DispatchTestCase


@torch.library.custom_op("torchtrt_ex::elementwise_mul_aot", mutates_args=())  # type: ignore[misc]
def elementwise_mul_aot(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return X * Y


@torch.library.register_fake("torchtrt_ex::elementwise_mul_aot")
def _(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(X)


_ELEMENTWISE_MUL_PTX = r"""
.version 8.0
.target sm_70
.address_size 64

.visible .entry elementwise_mul_aot_kernel(
    .param .u64 input_x,
    .param .u64 input_y,
    .param .u64 output_z
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [input_x];
    ld.param.u64 %rd2, [input_y];
    ld.param.u64 %rd3, [output_z];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r4, %r1, %r2, %r3;
    setp.ge.u32 %p1, %r4, 4096;
    @%p1 bra DONE;
    mul.wide.u32 %rd4, %r4, 4;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;
    add.s64 %rd7, %rd3, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    mul.rn.f32 %f3, %f1, %f2;
    st.global.f32 [%rd7], %f3;
DONE:
    ret;
}
"""


if platform.system() != "Windows" and torch_tensorrt.ENABLED_FEATURES.qdp_plugin:
    import tensorrt.plugin as trtp

    def elementwise_mul_autotune(
        X: trtp.TensorDesc,
        Y: trtp.TensorDesc,
        outputs: Tuple[trtp.TensorDesc],
    ) -> List[trtp.AutoTuneCombination]:
        return [trtp.AutoTuneCombination("FP32, FP32, FP32", "LINEAR", [1])]

    def elementwise_mul_aot_impl(
        X: trtp.TensorDesc,
        Y: trtp.TensorDesc,
        outputs: Tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Tuple[
        str,
        bytes,
        trtp.KernelLaunchParams,
        trtp.SymIntExprs,
    ]:
        launch = trtp.KernelLaunchParams()
        launch.grid_x = trtp.SymInt32(64)
        launch.grid_y = trtp.SymInt32(1)
        launch.grid_z = trtp.SymInt32(1)
        launch.block_x = 64
        launch.block_y = 1
        launch.block_z = 1
        launch.shared_mem = 0
        return (
            "elementwise_mul_aot_kernel",
            _ELEMENTWISE_MUL_PTX.encode("utf-8"),
            launch,
            trtp.SymIntExprs(0),
        )

    torch_tensorrt.dynamo.conversion.plugins.custom_op(
        "torchtrt_ex::elementwise_mul_aot",
        aot_impl=elementwise_mul_aot_impl,
        autotune=elementwise_mul_autotune,
    )


@unittest.skipIf(platform.system() == "Windows", "AOT QDP is not tested on Windows")
@unittest.skipIf(
    not torch_tensorrt.ENABLED_FEATURES.qdp_plugin,
    "QDP Plugin is not available",
)
class TestAutomaticPluginAot(DispatchTestCase):
    def test_aot_mul_plugin(self):
        class ElementwiseMul(nn.Module):
            def forward(self, lhs, rhs):
                return torch.ops.torchtrt_ex.elementwise_mul_aot.default(lhs, rhs)

        inputs = [
            torch.randn((64, 64), device="cuda", dtype=torch.float32),
            torch.randn((64, 64), device="cuda", dtype=torch.float32),
        ]
        self.run_test(ElementwiseMul(), inputs)


if __name__ == "__main__":
    run_tests()
