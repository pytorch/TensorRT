import torch
import torch_tensorrt
from transformers import AutoModelForCausalLM, AutoTokenizer


def export_llama2(model, inputs):
    """
    Exports the llama2 model into an ExportedProgram
    """
    with torch.no_grad():
        seq_len = torch.export.Dim("seq_len", min=1, max=64)
        ep = torch.export.export(
            model, (inputs,), dynamic_shapes=({1: seq_len},), strict=False
        )

    return ep


# Define the Llama2 model from hugging face
# kv_cache is not supported in Torch-TRT currently.
# attn_implementation=sdpa has tracing issues
llama_path = "meta-llama/Llama-2-7b-hf"
model = (
    AutoModelForCausalLM.from_pretrained(
        llama_path, use_cache=False, attn_implementation="eager"
    )
    .eval()
    .cuda()
)
tokenizer = AutoTokenizer.from_pretrained(llama_path)

base_prompt = "How many hours are in a day?"
base_inputs = tokenizer(base_prompt, return_tensors="pt").to("cuda:0")
input_ids = base_inputs.input_ids
pyt_out = model(input_ids)

llama2_ep = export_llama2(model, input_ids)
trt_model = torch_tensorrt.dynamo.compile(
    llama2_ep,
    inputs=[input_ids],
    enabled_precisions={torch.float32},
    min_block_size=1,
    truncate_double=True,
    torch_executed_ops={"torch.ops.aten.slice.Tensor"},
    debug=True,
    disable_tf32=True,
)

trt_out = trt_model(input_ids)
# breakpoint()
# print("Mean diff: ", torch.mean(torch.abs(pyt_out.logits-trt_out.logits)))
print("Mean diff: ", torch.mean(torch.abs(pyt_out - trt_out)))
breakpoint()
print("done")
