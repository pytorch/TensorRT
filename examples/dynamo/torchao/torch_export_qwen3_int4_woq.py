"""
.. _torch_export_qwen3_int4_woq:

Compiling Qwen3-8B with TorchAO INT4 weight-only quantization
=============================================================

This example loads
`pytorch/Qwen3-8B-INT4 <https://huggingface.co/pytorch/Qwen3-8B-INT4>`_,
converts Hub HQQ / tile-packed INT4 into TensorRT-compatible **symmetric**
group-wise Int4Tensor weights, exports a logits-only wrapper, and compiles
with the Torch-TensorRT Dynamo backend.

Hub checkpoints are often asymmetric and may use Int4TilePackedTo4dTensor.
Those layouts are dequantized to dense BF16, then re-packed with
quantize_linear_int4_symmetric so IDequantizeLayer sees a zero
zero-point and group-wise INT4 storage.

The compiled engine is fixed to SEQ_LEN. Greedy decode pads or windows the
growing sequence into that length (no KV cache), matching other Torch-TensorRT
LLM examples.

.. code-block:: bash

    pip install torchao transformers accelerate

    export QWEN3_INT4_MODEL=pytorch/Qwen3-8B-INT4
    python torch_export_qwen3_int4_woq.py

"""

# %%
# Imports
# -------
# This example lives in examples/dynamo/torchao/. Move that directory off
# the front of sys.path so import torchao resolves the PyPI package
# instead of this folder.

from __future__ import annotations

import copy
import gc
import os
import sys
from pathlib import Path

_EXAMPLE_DIR = str(Path(__file__).resolve().parent)
if sys.path and Path(sys.path[0]).resolve() == Path(_EXAMPLE_DIR):
    sys.path.pop(0)

import torch
import torch_tensorrt as torchtrt
from transformers import AutoConfig, AutoTokenizer, Qwen3ForCausalLM

sys.path.insert(0, _EXAMPLE_DIR)
from int4_utils import convert_hub_int4_to_symmetric_trt, pre_process_model_for_export
from utils import exclude_dq_from_constant_folding

MODEL_ID = os.environ.get("QWEN3_INT4_MODEL", "pytorch/Qwen3-8B-INT4")
MAX_NEW_TOKENS = int(os.environ.get("QWEN3_MAX_NEW_TOKENS", "64"))
GROUP_SIZE = int(os.environ.get("QWEN3_GROUP_SIZE", "128"))
PROMPT = os.environ.get(
    "QWEN3_PROMPT",
    "Explain weight-only INT4 quantization in one short paragraph.",
)
SEQ_LEN = int(os.environ.get("QWEN3_SEQ_LEN", "128"))

# %%
# Sanitize Hub TorchAO config and load the model
# ----------------------------------------------
# Older checkpoints serialize layout=TensorCoreTiledLayout and other kwargs
# removed in torchao>=0.17. Rewrite those onto int4_packing_format before
# from_pretrained.


def sanitize_torchao_quant_config(qc):
    if qc is None:
        return None
    if hasattr(qc, "to_dict"):
        d = qc.to_dict()
    elif isinstance(qc, dict):
        d = copy.deepcopy(qc)
    else:
        d = dict(qc)

    qt = d.get("quant_type")
    if isinstance(qt, dict) and "default" in qt:
        entry = qt["default"]
        data = entry.get("_data", entry)
        layout = data.pop("layout", None)
        if layout is not None and data.get("int4_packing_format") is None:
            data["int4_packing_format"] = "tile_packed_to_4d"
        if isinstance(layout, dict):
            inner = (layout.get("_data") or {}).get("inner_k_tiles")
            if inner is not None:
                data["int4_tile_packed_ntile"] = inner
        for k in ("use_hqq", "zero_point_domain", "preserve_zero", "layout"):
            data.pop(k, None)
        allowed = {
            "group_size",
            "set_inductor_config",
            "int4_packing_format",
            "int4_choose_qparams_algorithm",
            "int4_tile_packed_ntile",
        }
        entry["_data"] = {k: v for k, v in data.items() if k in allowed}
        qt["default"] = entry
        d["quant_type"] = qt
    return d


class _LogitsWrapper(torch.nn.Module):
    """CausalLM → logits only (export / TRT friendly)."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
        ).logits


def make_position_ids(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(seq_len, device=device).unsqueeze(0)


print(f"Loading {MODEL_ID} ...")
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=False)
config.quantization_config = sanitize_torchao_quant_config(config.quantization_config)
config.use_cache = False

model = (
    Qwen3ForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=False,
    )
    .cuda()
    .eval()
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# %%
# Convert Hub INT4 → symmetric Int4Tensor
# ---------------------------------------

convert_hub_int4_to_symmetric_trt(model, group_size=GROUP_SIZE, verbose=False)
pre_process_model_for_export(model)

# %%
# Tokenize a fixed-length prompt
# ------------------------------

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": PROMPT},
]
templated = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
encoded = tokenizer(
    templated,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=SEQ_LEN,
)
input_ids = encoded["input_ids"].cuda()
position_ids = make_position_ids(SEQ_LEN, input_ids.device)
wrapper = _LogitsWrapper(model).eval()

# %%
# Greedy decode without KV cache
# ------------------------------
# The compiled engine is fixed to SEQ_LEN, so each step pads or windows the
# growing sequence and reads logits at the last real (non-pad) position.


def greedy_generate(
    logits_fn,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int,
    pad_token_id: int,
) -> torch.Tensor:
    nonempty = (input_ids[0] != pad_token_id).nonzero(as_tuple=False)
    if nonempty.numel() == 0:
        cur = input_ids.clone()
    else:
        cur = input_ids[:, : int(nonempty[-1]) + 1].clone()

    for _ in range(max_new_tokens):
        if cur.shape[1] >= SEQ_LEN:
            window = cur[:, -SEQ_LEN:]
            real_len = SEQ_LEN
            start_pos = cur.shape[1] - SEQ_LEN
            step_position_ids = torch.arange(
                start_pos, start_pos + SEQ_LEN, device=cur.device
            ).unsqueeze(0)
        else:
            pad = torch.full(
                (cur.shape[0], SEQ_LEN - cur.shape[1]),
                pad_token_id,
                dtype=cur.dtype,
                device=cur.device,
            )
            window = torch.cat([cur, pad], dim=1)
            real_len = cur.shape[1]
            step_position_ids = make_position_ids(SEQ_LEN, cur.device)

        logits = logits_fn(window, step_position_ids)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        next_id = logits[:, real_len - 1, :].argmax(dim=-1, keepdim=True)
        cur = torch.cat([cur, next_id], dim=-1)
        if int(next_id.item()) == int(eos_token_id):
            break
    return cur


# %%
# Eager baseline
# --------------

with torch.no_grad():
    eager_logits = wrapper(input_ids, position_ids)
    eager_ids = greedy_generate(
        wrapper,
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
print("eager:", tokenizer.decode(eager_ids[0], skip_special_tokens=True))

# %%
# Export and compile
# ------------------
# strict=False matches other Torch-TensorRT LLM examples (aotautograd).
# immutable_weights=True is required for INT4 packed constants.

processed = pre_process_model_for_export(wrapper)
with exclude_dq_from_constant_folding():
    exp_program = torch.export.export(
        processed,
        (input_ids, position_ids),
        strict=False,
    )

compiled = torchtrt.dynamo.compile(
    exp_program,
    inputs=[input_ids, position_ids],
    truncate_double=True,
    require_full_compilation=True,
    immutable_weights=True,
    min_block_size=1,
    use_explicit_typing=True,
)

del exp_program, processed
gc.collect()
torch.cuda.empty_cache()

# %%
# Torch-TensorRT inference
# ------------------------

with torch.no_grad():
    trt_logits = compiled(input_ids, position_ids)
if isinstance(trt_logits, (tuple, list)):
    trt_logits = trt_logits[0]
trt_logits = trt_logits.to(device=eager_logits.device, dtype=eager_logits.dtype)

diff = (trt_logits.float() - eager_logits.float()).abs()
print(f"max |Δlogits|:  {diff.max().item():.6g}")
print(f"mean |Δlogits|: {diff.mean().item():.6g}")

nonempty = (input_ids[0] != tokenizer.pad_token_id).nonzero(as_tuple=False)
last_pos = int(nonempty[-1]) if nonempty.numel() else input_ids.shape[1] - 1
eager_tok = eager_logits[0, last_pos].argmax()
trt_tok = trt_logits[0, last_pos].argmax()
print(
    f"next-token argmax @ pos {last_pos}: "
    f"eager={int(eager_tok)} trt={int(trt_tok)} "
    f"match={bool(eager_tok == trt_tok)}"
)

with torch.no_grad():
    trt_ids = greedy_generate(
        compiled,
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
print("trt:", tokenizer.decode(trt_ids[0], skip_special_tokens=True))
