import torch
import torch_tensorrt
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
)


def export_llama2(model, inputs):
    """
    Exports the llama2 model into an ExportedProgram
    """
    with torch.no_grad():
        # max=1024 has contraint violation error. https://github.com/pytorch/pytorch/issues/125604
        seq_len = torch.export.Dim("seq_len", min=1, max=16)
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
llama_path = "meta-llama/Llama-2-7b-chat-hf"
model = (
    AutoModelForCausalLM.from_pretrained(
        llama_path, use_cache=False, attn_implementation="sdpa"
    )
    .eval()
    .cuda()
)
tokenizer = AutoTokenizer.from_pretrained(llama_path)

base_prompt = "Can you explain what is dynamic programming?"
base_inputs = tokenizer(base_prompt, return_tensors="pt").to("cuda:0")
input_ids = base_inputs.input_ids

max_tokens = 40
pyt_out = model(input_ids)
# generate_ids = model.generate(base_inputs.input_ids, max_length=max_tokens)

llama2_ep = export_llama2(model, input_ids)
trt_model = torch_tensorrt.dynamo.compile(
    llama2_ep,
    inputs=[input_ids],
    enabled_precisions={torch.float32},
    min_block_size=1,
    truncate_double=True,
    debug=True,
    disable_tf32=True,
)

trt_out = trt_model(input_ids)
# breakpoint()
# print("Mean diff: ", torch.mean(torch.abs(pyt_out.logits-trt_out.logits)))
print("Mean diff: ", torch.mean(torch.abs(pyt_out.logits - trt_out.logits)))
breakpoint()
# Auto-regressive generation loop for greedy search
max_length = len(input_ids) + max_tokens
stopping_criteria = StoppingCriteriaList(
    [
        MaxLengthCriteria(max_length=max_length),
        EosTokenCriteria(eos_token_id=tokenizer.eos_token_id),
    ]
)

token_id = 0
while token_id < max_tokens:
    print("Generating token: ", token_id)
    trt_outputs = model(input_ids)
    logits = trt_outputs.logits
    next_token_logits = logits[:, -1, :]
    next_tokens = torch.argmax(next_token_logits, dim=-1)
    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
    if stopping_criteria(input_ids, logits).item():
        break
    token_id += 1


# Decode the sentence
print("=============================")
# print(
#     "Pytorch model generated text: ",
#     tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0],
# )
print("=============================")
print(
    "TensorRT model generated text: ",
    tokenizer.batch_decode(
        input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0],
)
