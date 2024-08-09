import torch
import torch_tensorrt
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import export_llm, generate

# Define the parameters
MAX_TOKENS = 32
DEVICE = torch.device("cuda:0")

# Define the Llama2 model from hugging face
# kv_cache is not supported in Torch-TRT currently.
# CPU is used here so that GPU memory is reserved for TRT compilation.
llama_path = "meta-llama/Llama-2-7b-chat-hf"
with torch.no_grad():
    model = (
        AutoModelForCausalLM.from_pretrained(
            llama_path, use_cache=False, attn_implementation="eager"
        )
        .eval()
    )

tokenizer = AutoTokenizer.from_pretrained(llama_path)
base_prompt = "What is dynamic programming?"
base_inputs = tokenizer(base_prompt, return_tensors="pt")
input_ids = base_inputs.input_ids

# Auto-regressive generation loop for greedy search using PyTorch model
pyt_gen_tokens = generate(model, input_ids, MAX_TOKENS, tokenizer.eos_token_id)

# Export the llama2 model into an ExportedProgram which is input of TRT compilation
llama2_ep = export_llm(model, input_ids, max_seq_len=64)
trt_model = torch_tensorrt.dynamo.compile(
    llama2_ep,
    inputs=[input_ids],
    enabled_precisions={torch.float32},
    min_block_size=1,
    truncate_double=True,
    debug=True,
    device=DEVICE,
    disable_tf32=True,
)

# Auto-regressive generation loop for greedy search
# Move inputs to GPU
input_ids = input_ids.to(DEVICE)
generated_token_ids = generate(trt_model, input_ids, MAX_TOKENS, tokenizer.eos_token_id)

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
