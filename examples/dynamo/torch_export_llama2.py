import torch
import torch_tensorrt
from transformers import AutoModelForCausalLM, AutoTokenizer


def export_llama2(model, inputs):
    """
    Exports the llama2 model into an ExportedProgram
    """
    with torch.no_grad():
        # max=1024 has contraint violation error. https://github.com/pytorch/pytorch/issues/125604
        seq_len = torch.export.Dim("seq_len", min=1, max=1024)
        try:
            print("Trying to export the model using torch.export.export()..")
            # strict=False only enables aotautograd tracing and excludes dynamo.
            ep = torch.export.export(
                model, (inputs,), dynamic_shapes=({1: seq_len},), strict=False
            )
        except:
            print(
                "Trying torch.export._trace._export to trace the graph since torch.export.export() failed"
            )
            # This API is used to express the constraint violation guards as asserts in the graph.
            ep = torch.export._trace._export(
                model,
                (inputs,),
                dynamic_shapes=({1: seq_len},),
                strict=False,
                _allow_complex_guards_as_runtime_asserts=True,
            )

    return ep


# Define the Llama2 model from hugging face
# kv_cache is not supported in Torch-TRT currently.
# attn_implementation=sdpa has tracing issues
llama_path = "meta-llama/Llama-2-7b-hf"
model = (
    AutoModelForCausalLM.from_pretrained(
        llama_path, use_cache=False, attn_implementation="sdpa"
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
    # torch_executed_ops={"torch.ops.aten.slice.Tensor"},
    debug=True,
    disable_tf32=True,
)

trt_out = trt_model(input_ids)
breakpoint()
# print("Mean diff: ", torch.mean(torch.abs(pyt_out.logits-trt_out.logits)))
print("Mean diff: ", torch.mean(torch.abs(pyt_out - trt_out)))
# breakpoint()
# print("done")
