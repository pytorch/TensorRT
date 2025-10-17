import gc
import unittest
from math import ceil

import cuda.tile as ct
import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt._enums import dtype

from ...conversion.harness import DispatchTestCase
from .matmul import matmul_kernel


def clean_gpu_memory(message: str = ""):
    torch.cuda.empty_cache()
    gc.collect()
    print(f"completed cleaning gpu memory: {message}")


def print_gpu_memory(message: str = ""):
    remaining_memory, total_memory = torch.cuda.mem_get_info()
    print(f"{message} Remaining GPU memory: {remaining_memory / 1024 / 1024} MB")
    print(f"{message} Total GPU memory: {total_memory / 1024 / 1024} MB")
    print(
        f"{message} Used GPU memory {(total_memory - remaining_memory)/total_memory*100:.2f}%, Used GPU memory: {(total_memory - remaining_memory) / 1024 / 1024} MB"
    )


@torch.library.custom_op("cutile::matmul", mutates_args=())  # type: ignore[misc]
def cutile_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    tile_size_m: int = 256,
    tile_size_n: int = 256,
    tile_size_k: int = 64,
) -> torch.Tensor:
    C = torch.empty(A.shape[0], B.shape[1], device=A.device, dtype=A.dtype)
    tm, tn, tk = tile_size_m, tile_size_n, tile_size_k
    m, n, _ = A.shape[0], B.shape[1], A.shape[1]
    grid = (ceil(m / tm) * ceil(n / tn), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, matmul_kernel, (A, B, C, tm, tn, tk))
    return C


@torch.library.register_fake("cutile::matmul")
def _(
    A: torch.Tensor,
    B: torch.Tensor,
    tile_size_m: int = 256,
    tile_size_n: int = 256,
    tile_size_k: int = 64,
) -> torch.Tensor:
    return torch.empty(A.shape[0], B.shape[1], device=A.device, dtype=A.dtype)


if not torch_tensorrt.ENABLED_FEATURES.tensorrt_rtx:
    torch_tensorrt.dynamo.conversion.plugins.custom_op(
        "cutile::matmul", supports_dynamic_shapes=True
    )


@unittest.skipIf(
    torch.cuda.get_device_capability() < (10, 0),
    "cuTile requires compute capability 10.0 or later",
)
class TestAutomaticPlugin(DispatchTestCase):

    @parameterized.expand(
        [
            ((64, 64), (64, 128), torch.float16),
            # ((256, 256), (256, 16), torch.float16),
        ]
    )
    def test_matmul(self, a_shape, b_shape, data_type):
        class cutile_matmul(nn.Module):
            def forward(self, a, b):
                return torch.ops.cutile.matmul.default(a, b)

        class torch_matmul(nn.Module):
            def forward(self, a, b):
                return torch.matmul(a, b)

        inputs = (
            torch.randn(a_shape, device="cuda", dtype=data_type),
            torch.randn(b_shape, device="cuda", dtype=data_type),
        )
        enable_cutile, enable_trt_native = False, True
        if enable_cutile:
            with torch.no_grad():
                cutile_mod = cutile_matmul()
                print_gpu_memory("before cutile export")
                clean_gpu_memory("before cutile export")
                cutile_mod_ep = torch.export.export(cutile_mod, inputs)
                print_gpu_memory("after cutile export")
                clean_gpu_memory("after cutile export")
                with torch_tensorrt.dynamo.Debugger(
                    "graphs",
                    logging_dir="debuglogs_matmul",
                    capture_fx_graph_after=["constant_fold"],
                    save_engine_profile=True,
                    profile_format="trex",
                    engine_builder_monitor=False,
                ):
                    trt_cutile_mod = torch_tensorrt.dynamo.compile(
                        cutile_mod_ep,
                        inputs,
                        min_block_size=1,
                    )
                print_gpu_memory("after cutile compile")
                clean_gpu_memory("after cutile compile")

                outputs_cutile = trt_cutile_mod(*inputs)
                print_gpu_memory("after cutile inference")
                clean_gpu_memory("after cutile inference")
                print(f"outputs_cutile: {outputs_cutile}")

        if enable_trt_native:
            with torch.no_grad():
                torch_mod = torch_matmul()
                torch_mod_ep = torch.export.export(torch_mod, inputs)
                print_gpu_memory("after trt native export")
                clean_gpu_memory("after trt native export")
                with torch_tensorrt.dynamo.Debugger(
                    "graphs",
                    logging_dir="debuglogs_matmul",
                    capture_fx_graph_after=["constant_fold"],
                    save_engine_profile=True,
                    profile_format="trex",
                    engine_builder_monitor=False,
                ):
                    trt_torch_mod = torch_tensorrt.dynamo.compile(
                        torch_mod_ep,
                        inputs,
                        min_block_size=1,
                    )
                print_gpu_memory("after trt native compile")
                clean_gpu_memory("after trt native compile")
                outputs_trt = trt_torch_mod(*inputs)
                print_gpu_memory("after trt native inference")
                clean_gpu_memory("after trt native inference")
                print(f"outputs_trt: {outputs_trt}")

        if enable_trt_native and enable_cutile:
            self.assertTrue(
                torch.allclose(outputs_cutile, outputs_trt, atol=1e-4, rtol=1e-4)
            )


if __name__ == "__main__":
    run_tests()
