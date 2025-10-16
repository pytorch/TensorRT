import math
import unittest

import cuda.tile as ct
import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt._enums import dtype

from ...conversion.harness import DispatchTestCase
from .attention import fmha_kernel


@torch.library.custom_op("cutile::flash_attention", mutates_args=())  # type: ignore[misc]
def cutile_flash_attention(
    Q: torch.Tensor,  # (batch_size, q_heads, q_len, hidden_size)
    K: torch.Tensor,  # (batch_size, k_heads, k_len, hidden_size)
    V: torch.Tensor,  # (batch_size, k_heads, k_len, hidden_size)
    is_causal: bool = False,
    tile_size_m: int = 8,
    tile_size_n: int = 16,
) -> torch.Tensor:
    TILE_M, TILE_N = tile_size_m, tile_size_n
    batch_size, q_heads, q_len, hidden_size = Q.shape
    _, k_heads, k_len, _ = K.shape
    query_group_size = q_heads // k_heads
    qk_scale = 1 / math.sqrt(hidden_size)
    O = torch.zeros_like(Q)
    EVEN_K = (k_len % TILE_N) == 0
    grid = (math.ceil(q_len / TILE_M), batch_size * q_heads, 1)
    input_pos = (
        0  # TODO: figure out how to use the input_pos, for now do not use, set to 0
    )
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        fmha_kernel,
        (
            Q,
            K,
            V,
            O,
            qk_scale,
            input_pos,
            hidden_size,
            q_heads,
            TILE_M,
            TILE_N,
            query_group_size,
            is_causal,
            EVEN_K,
        ),
    )
    return O


@torch.library.register_fake("cutile::flash_attention")
def _(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal: bool = False,
    tile_size_m: int = 8,
    tile_size_n: int = 16,
) -> torch.Tensor:
    return Q


if not torch_tensorrt.ENABLED_FEATURES.tensorrt_rtx:
    torch_tensorrt.dynamo.conversion.plugins.custom_op(
        "cutile::flash_attention", supports_dynamic_shapes=True
    )


@unittest.skipIf(
    torch.cuda.get_device_capability() < (10, 0),
    "cuTile requires compute capability 10.0 or later",
)
class TestAutomaticPlugin(DispatchTestCase):

    @parameterized.expand(
        [
            (
                (32, 8, 128, 64),
                (32, 8, 128, 64),
                True,
            ),  # is_causal=True, will not use the mask
        ]
    )
    def test_flash_attention(self, query_shape, key_shape, is_causal):
        class cutile_flash_attention(nn.Module):
            def forward(self, Q, K, V):
                return torch.ops.cutile.flash_attention.default(Q, K, V, is_causal)

        class torch_flash_attention(nn.Module):
            def forward(self, Q, K, V, is_causal: bool = False):
                q_heads = query_shape[1]
                k_heads = key_shape[1]
                hidden_size = query_shape[3]
                qk_scale = 1 / math.sqrt(hidden_size)
                return torch.nn.functional.scaled_dot_product_attention(
                    Q,
                    K,
                    V,
                    is_causal=is_causal,
                    scale=qk_scale,
                    enable_gqa=(q_heads != k_heads),
                )

        inputs = (
            torch.randn(query_shape, device="cuda", dtype=torch.float16).cuda(),
            torch.randn(key_shape, device="cuda", dtype=torch.float16).cuda(),
            torch.randn(key_shape, device="cuda", dtype=torch.float16).cuda(),
        )

        cutile_mod = cutile_flash_attention()
        torch_mod = torch_flash_attention()
        cutile_mod_ep = torch.export.export(cutile_mod, inputs)
        torch_mod_ep = torch.export.export(torch_mod, inputs)
        trt_cutile_mod = torch_tensorrt.dynamo.compile(
            cutile_mod_ep,
            inputs,
            precision=dtype.f16,
        )
        trt_torch_mod = torch_tensorrt.dynamo.compile(
            torch_mod_ep,
            inputs,
            precision=dtype.f16,
        )
        with torch.no_grad():
            outputs_cutile = trt_cutile_mod(*inputs)
            outputs_trt = trt_torch_mod(*inputs)
            self.assertTrue(
                torch.allclose(outputs_cutile, outputs_trt, atol=1e-4, rtol=1e-4)
            )


if __name__ == "__main__":
    run_tests()
