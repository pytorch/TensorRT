import torch
import torch_tensorrt
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
)


def export_gpt2(model, inputs):
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


# Define tokenizer and model
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = (
    AutoModelForCausalLM.from_pretrained(
        "gpt2",
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )
    .eval()
    .to(torch_device)
)

# Input prompt
model_inputs = tokenizer("I enjoy walking with my cute dog", return_tensors="pt").to(
    torch_device
)
input_ids = model_inputs["input_ids"]
max_tokens = 40

# Pyt model outputs
greedy_output = model.generate(**model_inputs, max_new_tokens=max_tokens)
pyt_outputs = model(input_ids)

# Compile Torch-TRT model
gpt2_ep = export_gpt2(model, input_ids)
trt_model = torch_tensorrt.dynamo.compile(
    gpt2_ep,
    inputs=[input_ids],
    enabled_precisions={torch.float32},
    truncate_double=True,
    torch_executed_ops={"torch.ops.aten.slice.Tensor"},
    disable_tf32=True,
)

trt_outputs = trt_model(input_ids)

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
    trt_outputs = model(input_ids)
    logits = trt_outputs.logits
    next_token_logits = logits[:, -1, :]
    next_tokens = torch.argmax(next_token_logits, dim=-1)
    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
    if stopping_criteria(input_ids, logits).item():
        break
    token_id += 1

# Decode the sentence
print(
    "Pytorch model generated text: ",
    tokenizer.decode(greedy_output[0], skip_special_tokens=True),
)
print("=============================")
print(
    "TensorRT model generated text: ",
    tokenizer.decode(input_ids[0], skip_special_tokens=True),
)
