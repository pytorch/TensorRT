import importlib.util
import platform
import unittest
from math import ceil

import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized

if not importlib.util.find_spec("cuda.tile"):
    print("cuda.tile is not installed, skipping cuTile tests")
else:
    import cuda.tile as ct

    from .cutile.matmul import matmul_kernel

    def register_cutile_matmul():
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
            ct.launch(
                torch.cuda.current_stream(), grid, matmul_kernel, (A, B, C, tm, tn, tk)
            )
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

        torch_tensorrt.dynamo.conversion.plugins.custom_op(
            "cutile::matmul", supports_dynamic_shapes=True
        )

    @unittest.skipIf(
        torch.cuda.get_device_capability() < (10, 0),
        "cuTile requires compute capability 10.0 or later",
    )
    @unittest.skipIf(
        not importlib.util.find_spec("cuda.tile"),
        "cuda.tile is required to run this test",
    )
    @unittest.skipIf(
        platform.system() != "Linux",
        "cuTile is only supported on Linux for now",
    )
    @unittest.skipIf(
        torch_tensorrt.ENABLED_FEATURES.tensorrt_rtx,
        "TensorRT RTX does not support plugins which is required for cuTile",
    )
    class TestMatmul:
        register_cutile_matmul()

        @parameterized.expand(
            [
                ((64, 64), (64, 128), torch.float16),
                ((256, 256), (256, 16), torch.float16),
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
            enable_cutile, enable_trt_native = True, True
            if enable_cutile:
                with torch.no_grad():
                    cutile_mod = cutile_matmul()
                    cutile_mod_ep = torch.export.export(cutile_mod, inputs)
                    trt_cutile_mod = torch_tensorrt.dynamo.compile(
                        cutile_mod_ep,
                        inputs,
                        min_block_size=1,
                    )
                    outputs_cutile = trt_cutile_mod(*inputs)

            if enable_trt_native:
                with torch.no_grad():
                    torch_mod = torch_matmul()
                    torch_mod_ep = torch.export.export(torch_mod, inputs)

                    trt_torch_mod = torch_tensorrt.dynamo.compile(
                        torch_mod_ep,
                        inputs,
                        min_block_size=1,
                    )
                    outputs_trt = trt_torch_mod(*inputs)
                    print(f"outputs_trt: {outputs_trt.shape}")

            if enable_trt_native and enable_cutile:
                assert torch.allclose(outputs_cutile, outputs_trt, atol=1e-4, rtol=1e-4)
