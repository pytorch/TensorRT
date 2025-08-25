"""
.. _torch_compile_gpt2:

Compiling GPT2 using the Torch-TensorRT ``torch.compile`` frontend
==========================================================

This example illustrates the state of the art model `GPT2 <https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>`_ optimized using
``torch.compile`` frontend of Torch-TensorRT. Install the following dependencies before compilation

.. code-block:: python

    pip install -r requirements.txt

GPT2 is a causal (unidirectional) transformer pretrained using language modeling on a very large corpus of text data. In this example, we use the GPT2 model available at `HuggingFace <https://huggingface.co/docs/transformers/en/model_doc/gpt2>`_ and apply torch.compile on it to
get the graph module representation of the graph. Torch-TensorRT converts this graph into an optimized TensorRT engine.
"""

# %%
# Import necessary libraries
# -----------------------------
import torch
import torch_tensorrt
from transformers import AutoModelForCausalLM, AutoTokenizer

# %%
# Define the necessary parameters
# -----------------------------
# Torch-TensorRT requires a GPU for successful compilation of the model.
# ``MAX_LENGTH`` is the maximum length the generated tokens can have. This corresponds to the length of the input prompt +
# number of new tokens generated
MAX_LENGTH = 32
DEVICE = torch.device("cuda:0")

# %%
# Model definition
# -----------------------------
# We use ``AutoModelForCausalLM`` class to load the pretrained GPT2 model from hugging face. ``kv_cache`` is not supported in Torch-TRT currently so ``use_cache=False``
with torch.no_grad():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = (
        AutoModelForCausalLM.from_pretrained(
            "gpt2",
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
            attn_implementation="eager",
        )
        .to(DEVICE)
        .eval()
    )

# %%
# PyTorch inference
# -----------------------------
# Tokenize a sample input prompt and get pytorch model outputs
prompt = "I enjoy walking with my cute dog"
model_inputs = tokenizer(prompt, return_tensors="pt")
input_ids = model_inputs["input_ids"].to(DEVICE)

# %%
# The ``generate()`` API of the ``AutoModelForCausalLM`` class is used for auto-regressive generation with greedy decoding.
with torch.no_grad():
    pyt_gen_tokens = model.generate(
        input_ids,
        max_length=MAX_LENGTH,
        use_cache=False,
        pad_token_id=tokenizer.eos_token_id,
    )

# %%
# Torch-TensorRT compilation and inference
# -----------------------------
# The input sequence length is dynamic, so we mark it using ``torch._dynamo.mark_dynamic`` API.
# We provide a (min, max) range of this value so that TensorRT knows in advance what values to optimize for.
# Usually, this would be the context length for the model. We start with ``min=2`` due to the `0/1 specialization <https://docs.google.com/document/d/16VPOa3d-Liikf48teAOmxLc92rgvJdfosIy-yoT38Io/edit?fbclid=IwAR3HNwmmexcitV0pbZm_x1a4ykdXZ9th_eJWK-3hBtVgKnrkmemz6Pm5jRQ&tab=t.0#heading=h.ez923tomjvyk>`_
torch._dynamo.mark_dynamic(input_ids, 1, min=2, max=1023)
model.forward = torch.compile(
    model.forward,
    backend="tensorrt",
    dynamic=None,
    options={
        "enabled_precisions": {torch.float32},
        "disable_tf32": True,
        "min_block_size": 1,
    },
)

# %%
# Auto-regressive generation loop for greedy decoding using TensorRT model
# The first token generation compiles the model using TensorRT and the second token
# encounters recompilation (which is an issue currently that would be resolved in the future)
with torch.no_grad():
    trt_gen_tokens = model.generate(
        inputs=input_ids,
        max_length=MAX_LENGTH,
        use_cache=False,
        pad_token_id=tokenizer.eos_token_id,
    )

# %%
# Decode the output sentences of PyTorch and TensorRT
# -----------------------------
print(
    "Pytorch model generated text: ",
    tokenizer.decode(pyt_gen_tokens[0], skip_special_tokens=True),
)
print("=============================")
print(
    "TensorRT model generated text: ",
    tokenizer.decode(trt_gen_tokens[0], skip_special_tokens=True),
)

# %%
# The output sentences should look like

"""
Pytorch model generated text:  I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to walk with my dog. I'm not sure if I'll
=============================
TensorRT model generated text:  I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to walk with my dog. I'm not sure if I'll
"""
