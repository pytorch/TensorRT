import torch
import torch_tensorrt
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import export_llm, generate

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
gpt2_ep = export_llm(model, input_ids, max_seq_len=1024)
trt_model = torch_tensorrt.dynamo.compile(
    gpt2_ep,
    inputs=[input_ids],
    enabled_precisions={torch.float32},
    truncate_double=True,
    debug=True,
)

# Auto-regressive generation loop for greedy search using Torch-TensorRT model
generated_token_ids = generate(trt_model, input_ids, max_tokens, tokenizer.eos_token_id)

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
