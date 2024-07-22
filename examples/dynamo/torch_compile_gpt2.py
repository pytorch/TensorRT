import torch
import torch_tensorrt
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import generate

# Define tokenizer and model
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = (
    AutoModelForCausalLM.from_pretrained(
        "gpt2",
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False,
        attn_implementation="eager",
    )
    .eval()
    .to(torch_device)
)

# Input prompt
model_inputs = tokenizer("I enjoy walking with my cute dog", return_tensors="pt").to(
    torch_device
)
input_ids = model_inputs["input_ids"]
max_tokens = 20

# Auto-regressive generation loop for greedy search using PyTorch model
pyt_gen_tokens = generate(model, input_ids, max_tokens, tokenizer.eos_token_id)

# Compile Torch-TRT model
torch._dynamo.mark_dynamic(input_ids, 1, min=7, max=1023)
model.forward = torch.compile(
    model.forward,
    backend="tensorrt",
    dynamic=None,
    options={
        "enabled_precisions": {torch.float},
        "debug": True,
        "disable_tf32": True,
    },
)

# Auto-regressive generation loop for greedy search using Torch-TensorRT model
generated_token_ids = generate(model, input_ids, max_tokens, tokenizer.eos_token_id)

# Decode the sentence
print("=============================")
print(
    "Pytorch model generated text: ",
    tokenizer.decode(pyt_gen_tokens[0], skip_special_tokens=True),
)
print("=============================")
print(
    "TensorRT model generated text: ",
    tokenizer.decode(generated_token_ids[0], skip_special_tokens=True),
)
