import torch
import torch_tensorrt
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
)

# Define tokenizer and model
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = (
    AutoModelForCausalLM.from_pretrained(
        "gpt2", pad_token_id=tokenizer.eos_token_id, use_cache=False
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
print(
    "Pytorch model generated text: ",
    tokenizer.decode(greedy_output[0], skip_special_tokens=True),
)

# Compile Torch-TRT model
torch._dynamo.mark_dynamic(input_ids, 1, min=7, max=1023)
model.forward = torch.compile(
    model.forward,
    backend="tensorrt",
    dynamic=None,
    options={
        "enabled_precisions": {torch.float},
        "torch_executed_ops": {"torch.ops.aten.slice.Tensor"},
        "debug": True,
        "disable_tf32": True,
    },
)

# Auto-regressive generation loop for greedy search
stopping_criteria = StoppingCriteriaList(
    [
        MaxLengthCriteria(max_length=max_tokens),
        EosTokenCriteria(eos_token_id=tokenizer.eos_token_id),
    ]
)
token_id = 0
while token_id < 20:
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
    "TensorRT model generated text: ",
    tokenizer.decode(input_ids[0], skip_special_tokens=True),
)
