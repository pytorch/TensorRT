# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import platform
import unittest

import pytest
import torch
import torch_tensorrt
from torch_tensorrt._utils import is_tensorrt_rtx_version_supported

if importlib.util.find_spec("transformers"):
    import transformers


@pytest.mark.unit
@pytest.mark.critical
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.float32],
)
@pytest.mark.parametrize("decompose_attention", [True, False])
@unittest.skipIf(
    not importlib.util.find_spec("transformers"),
    "transformers is required to run this test",
)
@unittest.skipIf(
    platform.system().lower().startswith("windows"),
    "dynamic head dimension HF GQA test is not supported on Windows",
)
def test_dynamic_head_dim_with_hf_model(dtype, decompose_attention):
    if (
        torch_tensorrt.ENABLED_FEATURES.tensorrt_rtx
        and not is_tensorrt_rtx_version_supported("1.6")
    ):
        pytest.skip(
            "TensorRT-RTX >= 1.6 is required because it fixed some accuracy issues for FP16"
        )
        return

    from transformers import AutoModelForCausalLM

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    model = (
        AutoModelForCausalLM.from_pretrained(
            model_name, use_cache=False, attn_implementation="sdpa"
        )
        .eval()
        .cuda()
        .to(dtype)
    )

    generator = torch.Generator(device="cuda").manual_seed(15)
    input_ids = torch.randint(
        1,
        10000,
        (1, 64),
        dtype=torch.int64,
        device="cuda",
        generator=generator,
    )
    position_ids = torch.arange(64).unsqueeze(0).cuda()

    seq_len = torch.export.Dim("seq_len", min=1, max=128)
    try:
        ep = torch.export.export(
            model,
            args=(input_ids,),
            kwargs={"position_ids": position_ids},
            dynamic_shapes=({1: seq_len}, {1: seq_len}),
            strict=False,
        )
    except Exception:
        ep = torch.export._trace._export(
            model,
            args=(input_ids,),
            kwargs={"position_ids": position_ids},
            dynamic_shapes=({1: seq_len}, {1: seq_len}),
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )

    trt_model = torch_tensorrt.dynamo.compile(
        ep,
        inputs=[input_ids, position_ids],
        use_fp32_acc=True,
        device=torch.device("cuda:0"),
        min_block_size=1,
        decompose_attention=decompose_attention,
    )

    with torch.no_grad():
        ref = model(input_ids, position_ids=position_ids).logits
        out = trt_model(input_ids, position_ids)
        # TRT model may return CausalLMOutputWithPast or tuple
        if hasattr(out, "logits"):
            out = out.logits
        elif isinstance(out, (tuple, list)):
            out = out[0]

    if dtype == torch.float16:
        # fp16 logits over a 152K vocab: a handful of near-zero logits can land just
        # past atol on some GPUs (seen on Ampere: 1-2 of 9.7M at ~0.41). Keep the
        # per-element tolerance, bound the outlier count, and require TRT's predicted
        # token to be among the reference top-5 at every position.
        mismatched = ~torch.isclose(ref, out, rtol=1e-1, atol=3e-1)
        assert (
            mismatched.sum() <= 10
        ), f"{int(mismatched.sum())} fp16 logits out of tolerance"
        top5 = ref.topk(5, dim=-1).indices
        assert (
            (top5 == out.argmax(-1, keepdim=True)).any(-1).all()
        ), "TRT predicted token not in reference top-5"
    else:
        torch.testing.assert_close(ref, out, rtol=1e-1, atol=3e-1)
