import torch
import torch_tensorrt
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import export_llm, generate

# Define the Llama2 model from hugging face
# kv_cache is not supported in Torch-TRT currently.
# attn_implementation=sdpa has tracing issues
llama_path = "meta-llama/Llama-2-7b-chat-hf"
model = (
    AutoModelForCausalLM.from_pretrained(
        llama_path, use_cache=False, attn_implementation="eager"
    )
    .eval()
    .cuda()
)
tokenizer = AutoTokenizer.from_pretrained(llama_path)

base_prompt = "What is dynamic programming?"
base_inputs = tokenizer(base_prompt, return_tensors="pt").to("cuda:0")
input_ids = base_inputs.input_ids

max_tokens = 32
pyt_out = model(input_ids)

# Auto-regressive generation loop for greedy search using PyTorch model
pyt_gen_tokens = generate(model, input_ids, max_tokens, tokenizer.eos_token_id)

llama2_ep = export_llm(model, input_ids, max_seq_len=64)
trt_model = torch_tensorrt.dynamo.compile(
    llama2_ep,
    inputs=[input_ids],
    enabled_precisions={torch.float32},
    min_block_size=1,
    truncate_double=True,
    debug=True,
    use_python_runtime=True,
    disable_tf32=True,
)

trt_out = trt_model(input_ids)

# Auto-regressive generation loop for greedy search
generated_token_ids = generate(trt_model, input_ids, max_tokens, tokenizer.eos_token_id)

# Check output difference
print("Mean diff: ", torch.mean(torch.abs(pyt_out.logits - trt_out.logits)))

# Decode the sentence
print("=============================")
print(
    "Pytorch model generated text: ",
    tokenizer.batch_decode(
        pyt_gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0],
)
print("=============================")
print(
    "TensorRT model generated text: ",
    tokenizer.batch_decode(
        generated_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0],
)
