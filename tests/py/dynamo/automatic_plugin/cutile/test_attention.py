import math
import unittest

import cuda.tile as ct
import torch
import torch.nn as nn
import torch_tensorrt
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

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


class cutile_flash_attention(nn.Module):
    def forward(self, Q, K, V):
        return torch.ops.cutile.flash_attention.default(Q, K, V, True)


class torch_flash_attention(nn.Module):
    def forward(self, Q, K, V):
        return torch.nn.functional.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )


@unittest.skipIf(
    torch.cuda.get_device_capability() < (10, 0),
    "cuTile requires compute capability 10.0 or later",
)
class TestAttention(DispatchTestCase):
    def test_cutile_flash_attention(self):
        data_type = torch.float32
        inputs = (
            torch.randn((32, 8, 128, 64), device="cuda", dtype=data_type).cuda(),
            torch.randn((32, 8, 128, 64), device="cuda", dtype=data_type).cuda(),
            torch.randn((32, 8, 128, 64), device="cuda", dtype=data_type).cuda(),
        )
        enable_cutile, enable_trt_native = True, True
        if enable_cutile:
            with torch.no_grad():
                cutile_mod = cutile_flash_attention()
                cutile_mod_ep = torch.export.export(cutile_mod, inputs)
                with torch_tensorrt.dynamo.Debugger(
                    "graphs",
                    logging_dir="debuglogs_cutile_attention",
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
                outputs_cutile = trt_cutile_mod(*inputs)

        if enable_trt_native:
            with torch.no_grad():
                torch_mod = torch_flash_attention()
                torch_mod_ep = torch.export.export(torch_mod, inputs)
                with torch_tensorrt.dynamo.Debugger(
                    "graphs",
                    logging_dir="debuglogs_trt_native_attention",
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
                outputs_trt = trt_torch_mod(*inputs)

        if enable_cutile and enable_trt_native:
            self.assertTrue(
                torch.allclose(outputs_cutile, outputs_trt, atol=5e-3, rtol=1e-2)
            )

    def test_flash_attention_dynamic_shape(self):

        data_type = torch.float32
        input_specs = [
            Input(
                min_shape=(32, 8, 1, 64),
                opt_shape=(32, 8, 128, 64),
                max_shape=(32, 8, 256, 64),
                dtype=data_type,
            ),
            Input(
                min_shape=(32, 8, 1, 64),
                opt_shape=(32, 8, 128, 64),
                max_shape=(32, 8, 256, 64),
                dtype=data_type,
            ),
            Input(
                min_shape=(32, 8, 1, 64),
                opt_shape=(32, 8, 128, 64),
                max_shape=(32, 8, 256, 64),
                dtype=data_type,
            ),
        ]

        compile_inputs = (
            torch.randn((32, 8, 128, 64), device="cuda", dtype=data_type).cuda(),
            torch.randn((32, 8, 128, 64), device="cuda", dtype=data_type).cuda(),
            torch.randn((32, 8, 128, 64), device="cuda", dtype=data_type).cuda(),
        )
        inference_inputs = (
            torch.randn((32, 8, 256, 64), device="cuda", dtype=data_type).cuda(),
            torch.randn((32, 8, 256, 64), device="cuda", dtype=data_type).cuda(),
            torch.randn((32, 8, 256, 64), device="cuda", dtype=data_type).cuda(),
        )

        enable_cutile, enable_trt_native = True, True
        q_len_dim = torch.export.Dim("q_len", min=1, max=256)
        dynamic_shapes = {"Q": {2: q_len_dim}, "K": {2: q_len_dim}, "V": {2: q_len_dim}}
        if enable_cutile:
            with torch.no_grad():
                cutile_mod = cutile_flash_attention()
                cutile_mod_ep = torch.export.export(
                    cutile_mod,
                    compile_inputs,
                    dynamic_shapes=dynamic_shapes,
                    strict=False,
                )
                with torch_tensorrt.dynamo.Debugger(
                    "graphs",
                    logging_dir="debuglogs_cutile_attention",
                    capture_fx_graph_after=["constant_fold"],
                    save_engine_profile=True,
                    profile_format="trex",
                    engine_builder_monitor=False,
                ):
                    trt_cutile_mod = torch_tensorrt.dynamo.compile(
                        cutile_mod_ep,
                        input_specs,
                        min_block_size=1,
                        enable_precisions={data_type},
                    )
                outputs_cutile = trt_cutile_mod(*inference_inputs)

        if enable_trt_native:
            with torch.no_grad():
                torch_mod = torch_flash_attention()
                torch_mod_ep = torch.export.export(
                    torch_mod,
                    compile_inputs,
                    dynamic_shapes=dynamic_shapes,
                    strict=False,
                )
                with torch_tensorrt.dynamo.Debugger(
                    "graphs",
                    logging_dir="debuglogs_trt_native_attention",
                    capture_fx_graph_after=["constant_fold"],
                    save_engine_profile=True,
                    profile_format="trex",
                    engine_builder_monitor=False,
                ):
                    trt_torch_mod = torch_tensorrt.dynamo.compile(
                        torch_mod_ep,
                        input_specs,
                        min_block_size=1,
                        enable_precisions={data_type},
                    )
                outputs_trt = trt_torch_mod(*inference_inputs)

        if enable_cutile and enable_trt_native:
            self.assertTrue(
                torch.allclose(outputs_cutile, outputs_trt, atol=5e-3, rtol=1e-2)
            )


if __name__ == "__main__":
    run_tests()
