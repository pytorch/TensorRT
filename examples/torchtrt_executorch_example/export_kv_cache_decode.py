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

``--zero_copy`` exports the same model with the engine writing the cache buffer
directly, instead of ExecuTorch staging a copy for the delegate and copying the
result back. The persistence check is the same, and it is the check that matters
here: zero-copy removes the copy that was making the update stick, so if the
engine's in-place write is not reaching the caller's buffer the run fails. What
it cannot see is a ``--zero_copy`` export that degenerated into an ordinary
staged ``.pte`` -- the two are indistinguishable to it -- so
``check_zero_copy_kv`` refuses to write one.

Prerequisites
-------------
Install Torch-TensorRT with the ExecuTorch extra before running this example::

    pip install -e ".[executorch]"
"""

import argparse
import os

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


def _save_zero_copy(trt_gm: torch.fx.GraphModule, inputs: tuple, path: str) -> None:
    """Save a .pte whose engine updates the KV cache in place.

    Zero-copy needs both ends of the Edge boundary: ``zero_copy_kv`` before
    lowering, and ``zero_copy_backend_config`` on the config the program is
    finalized with. Without the second the cache is still staged and its updates
    are dropped, with no error. ``torch_tensorrt.save(output_format="executorch",
    zero_copy_kv=True)`` owns both steps and is the shorter way to the same .pte;
    this spells them out because both halves are shown, and because a program
    that reaches ``to_executorch()`` by any other route has to install the config
    itself.
    """
    from torch_tensorrt.executorch import (
        check_zero_copy_kv,
        export,
        zero_copy_backend_config,
    )

    # retrace=True here, retrace=False for the plain save() below, so the two
    # exporters are both covered. Which way round matters: the legacy exporter
    # declares the aliased KV outputs while building the program, so on that lane
    # export()'s declaration pass reads each engine's aliased_io only to find the
    # mutations already declared. A retraced program arrives undeclared, so this
    # is the lane where that read decides anything -- where an engine-aliased
    # cache is told from an ordinary copy-back buffer.
    edge = export(trt_gm, arg_inputs=inputs, retrace=True, zero_copy_kv=True)
    program = edge.to_executorch(zero_copy_backend_config())
    check_zero_copy_kv(program)
    with open(path, "wb") as output:
        program.write_to_file(output)
    if program._tensor_data:
        # A delegate carrying external weights (the CUDA backend does) keeps them
        # outside the .pte, and write_to_file does not persist them; without the
        # .ptd next to it the .pte cannot load. This model has none, but a model
        # built from this one may.
        program.write_tensor_data_to_file(os.path.dirname(os.path.abspath(path)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", default="kv_cache_decode.pte", help="Path to save the .pte"
    )
    parser.add_argument(
        "--zero_copy",
        action="store_true",
        help="Let the engine write the KV cache in place (elides the aliased "
        "delegate outputs; needs a runtime that supports them).",
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
        if args.zero_copy:
            _save_zero_copy(trt_gm, (tokens, input_pos), args.model_path)
        else:
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
