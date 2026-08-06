"""
.. _executorch_export_kv_cache:

Exporting a Caller-Owned KV-Cache Decode Model to ExecuTorch (.pte)
==================================================================

This example exports a minimal single-layer attention decode step whose KV cache
is a registered buffer updated in place with ``index_copy_``. Torch-TensorRT
carries the cache as a *caller-owned* mutable buffer through the ExecuTorch
delegate, so the engine's aliased KV output is bound in place to the caller's
buffer and persists across ``execute()`` calls.

The companion ``kv_cache_decode_check`` reference runner loads the resulting
``.pte`` and asserts that a decode step observes the KV a previous step wrote
(i.e. the cache is shared across ``execute()`` calls).

Prerequisites
-------------
Install Torch-TensorRT with the ExecuTorch extra before running this example::

    pip install -e ".[executorch]"
"""

import argparse

import torch
import torch_tensorrt

VOCAB = 64
DIM = 32
HEADS = 2
HEAD_DIM = 16
MAX_LEN = 16


class KVDecodeStep(torch.nn.Module):
    """One attention layer with an in-place (index_copy_) KV cache.

    ``forward(tokens[1,1], input_pos[1]) -> logits[1,1,VOCAB]``. The ``k_cache`` /
    ``v_cache`` buffers are written at ``input_pos`` and attended over up to
    ``input_pos`` (causal), so a later step's output depends on earlier steps'
    writes -- which only holds if the cache persists across ``execute()`` calls.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, DIM)
        self.pos_embed = torch.nn.Embedding(MAX_LEN, DIM)
        self.q = torch.nn.Linear(DIM, HEADS * HEAD_DIM, bias=False)
        self.k = torch.nn.Linear(DIM, HEADS * HEAD_DIM, bias=False)
        self.v = torch.nn.Linear(DIM, HEADS * HEAD_DIM, bias=False)
        self.o = torch.nn.Linear(HEADS * HEAD_DIM, DIM, bias=False)
        self.lm = torch.nn.Linear(DIM, VOCAB, bias=False)
        self.register_buffer("k_cache", torch.zeros(1, HEADS, MAX_LEN, HEAD_DIM))
        self.register_buffer("v_cache", torch.zeros(1, HEADS, MAX_LEN, HEAD_DIM))

    def forward(self, tokens: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        pos_idx = input_pos.reshape(-1)
        pos = input_pos.reshape(())
        x = self.embed(tokens) + self.pos_embed(input_pos.reshape(1, 1))

        def split_heads(proj: torch.Tensor) -> torch.Tensor:
            return proj.view(1, 1, HEADS, HEAD_DIM).transpose(1, 2)

        q = split_heads(self.q(x))
        k = split_heads(self.k(x))
        v = split_heads(self.v(x))

        self.k_cache.index_copy_(2, pos_idx, k)
        self.v_cache.index_copy_(2, pos_idx, v)

        scores = (q @ self.k_cache.transpose(-1, -2)) / (HEAD_DIM**0.5)
        allowed = torch.arange(MAX_LEN, device=x.device) <= pos
        bias = torch.where(
            allowed,
            torch.zeros((), dtype=x.dtype, device=x.device),
            torch.full((), torch.finfo(x.dtype).min, dtype=x.dtype, device=x.device),
        )
        attn = torch.softmax(scores + bias.view(1, 1, 1, MAX_LEN), dim=-1)
        out = (attn @ self.v_cache).transpose(1, 2).reshape(1, 1, HEADS * HEAD_DIM)
        return self.lm(self.o(out))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", default="kv_cache_decode.pte", help="Path to save the .pte"
    )
    args = parser.parse_args()

    with torch.no_grad():
        torch.manual_seed(0)
        model = KVDecodeStep().eval().cuda()
        tokens = torch.zeros(1, 1, dtype=torch.long).cuda()
        input_pos = torch.tensor([0], dtype=torch.long).cuda()

        exported_program = torch.export.export(model, (tokens, input_pos))
        trt_gm = torch_tensorrt.dynamo.compile(
            exported_program,
            arg_inputs=(tokens, input_pos),
            min_block_size=1,
            truncate_double=True,
        )
        torch_tensorrt.save(
            trt_gm,
            args.model_path,
            output_format="executorch",
            arg_inputs=(tokens, input_pos),
            retrace=False,
        )
        print(f"Saved {args.model_path} successfully.")


if __name__ == "__main__":
    main()
