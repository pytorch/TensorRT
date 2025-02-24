"""
.. _torch_export_gpt2:

Compiling GPT2 using the dynamo backend
==========================================================

This script illustrates Torch-TensorRT workflow with dynamo backend on popular GPT2 model.
"""

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
import torch
import torch_tensorrt
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
)
from utils import export_llm, generate

# %%

# Define the parameters and initialize the model
MAX_TOKENS = 6
DEVICE = torch.device("cuda:0")

# Define the GPT2 model from hugging face
# kv_cache is not supported in Torch-TRT currently.
# CPU is used here so that GPU memory is reserved for TRT compilation.
with torch.no_grad():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = (
        AutoModelForCausalLM.from_pretrained(
            "gpt2",
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
            attn_implementation="sdpa",
            num_hidden_layers=4,
        )
        .eval()
        .half()
        .to(DEVICE)
    )
# %%
# Tokenize a sample input prompt and get pytorch model outputs
prompt = "What is parallel programming ?"
model_inputs = tokenizer(prompt, return_tensors="pt")
input_ids = model_inputs["input_ids"].to(DEVICE)

# Auto-regressive generation loop for greedy decoding using PyTorch model
# We use a custom generate function which is very similar to the huggingface one.
pyt_gen_tokens = generate(model, input_ids, MAX_TOKENS, tokenizer.eos_token_id)


# Initialize kvcaching
# construct a cache info object
from torch_tensorrt.dynamo.lowering import AttentionRegistry, SequenceInfo

MAX_SEQ_LEN = input_ids.shape[-1] + MAX_TOKENS
sequence_info = SequenceInfo(
    max_seq_len=MAX_SEQ_LEN,
    max_batch_size=1,
    page_size=MAX_SEQ_LEN,
)

csi = torch_tensorrt.dynamo.lowering.CachedSequenceInterface(
    AttentionRegistry.get("FlashInfer"), sequence_info=sequence_info, device=DEVICE
)


# %%
# Compilation with `Torch-TensorRT` using dynamo backend and generate TensorRT outputs
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Export the GPT2 model into an ExportedProgram which is input of TRT compilation
# To compile the model in FP16, we do the following
# 1) Cast the model to FP16 via model.half()
# 2) Enable use_explicit_typing=True. Certain layers are explicitly casted to FP32 within the pytorch model and this flag respects this behavior during TRT compilation
# 3) Enable use_fp32_acc=True. This ensures all the matmuls are accumulated in FP32 precision (similar to PyTorch)
gpt2_ep = export_llm(model, input_ids, max_seq_len=1024)
with torch_tensorrt.logging.debug():
    trt_model = torch_tensorrt.dynamo.compile(
        gpt2_ep,
        inputs=[input_ids],
        enabled_precisions={torch.float32},
        truncate_double=True,
        device=DEVICE,
        disable_tf32=True,
        use_explicit_typing=True,
        use_fp32_acc=True,
        debug=True,
        insert_flashinfer_ops=True,
        cached_seq_interface=csi,
        min_block_size=1e7,
    )


def custom_generate(model, input_seq, csi, max_tokens, eos_token_id):
    """
    Greedy decoding of the model. This generates up to max_tokens.
    """
    # Max length of output seq = current input_seq length + max_tokens allowed to generate
    max_output_seq_length = input_seq.shape[1] + max_tokens
    stopping_criteria = StoppingCriteriaList(
        [
            MaxLengthCriteria(max_length=max_output_seq_length),
            EosTokenCriteria(eos_token_id=eos_token_id),
        ]
    )
    csi.info.reset()
    csi.info.nest_sequences(input_seq)

    while True:
        outputs = model(*csi.args)
        logits = outputs.logits
        next_token_logits = logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        input_seq = torch.cat([input_seq, next_tokens[:, None]], dim=-1)
        # TODO: Handle batch in this check
        if stopping_criteria(input_seq, logits).item():
            break

    return input_seq


trt_gen_tokens = custom_generate(
    trt_model, input_ids, csi, MAX_TOKENS, tokenizer.eos_token_id
)
# breakpoint()
# Auto-regressive generation loop for greedy decoding using TensorRT model
# We use a custom generate function which is very similar to the huggingface one.
# Move inputs to GPU
# input_ids = input_ids.to(DEVICE)
# trt_gen_tokens = generate(trt_model, input_ids, MAX_TOKENS, tokenizer.eos_token_id)

# %%
# Decode the output sentences of PyTorch and TensorRT
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
print("=============================")
print(
    "Pytorch model generated text: ",
    tokenizer.decode(pyt_gen_tokens[0], skip_special_tokens=True),
)
print("=============================")
print(
    "TensorRT model generated text: ",
    tokenizer.decode(trt_gen_tokens[0], skip_special_tokens=True),
)

# Prompt : What is parallel programming ?

# =============================
# Pytorch model generated text: The parallel programming paradigm is a set of programming languages that are designed to be used in parallel. The main difference between parallel programming and parallel programming is that

# =============================
# TensorRT model generated text: The parallel programming paradigm is a set of programming languages that are designed to be used in parallel. The main difference between parallel programming and parallel programming is that
