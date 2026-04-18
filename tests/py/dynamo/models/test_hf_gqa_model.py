import pytest
import torch
import torch_tensorrt
from transformers import AutoModelForCausalLM


@pytest.mark.unit
@pytest.mark.critical
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            torch.float16,
            marks=pytest.mark.skip(
                reason="skip fp16 for now due to TRT's numeric diff"
            ),
        ),
        torch.float32,
    ],
)
@pytest.mark.parametrize("decompose_attention", [True, False])
def test_dynamic_head_dim_with_hf_model(dtype, decompose_attention):
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    model = (
        AutoModelForCausalLM.from_pretrained(
            model_name, use_cache=False, attn_implementation="sdpa"
        )
        .eval()
        .cuda()
        .to(dtype)
    )

    input_ids = torch.randint(1, 10000, (1, 64), dtype=torch.int64, device="cuda")
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
        use_explicit_typing=True,
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

    torch.testing.assert_close(ref, out, rtol=1e-1, atol=2e-1)
