.. _compile_hf_models:

Compiling LLM models from Huggingface
======================================

This tutorial walks you through how to compile LLM models from Huggingface using Torch-TensorRT. We also introduce KV caching in Torch-TensorRT which can greatly improve the performance of LLM inference. 
The code is available in the `tools/llm <https://github.com/pytorch/TensorRT/tree/main/tools/llm>`_ directory. We use the ``run_llm.py`` script to compile the model, generate outputs, and measure the performance.

.. note::
   This is an **experimental release** and APIs may change in future versions.

.. note::
   The compilation scripts and tutorials for Llama-2-7b-chat-hf and gpt2 models have been consolidated into the unified ``run_llm.py`` script located in the `tools/llm <https://github.com/pytorch/TensorRT/tree/main/tools/llm>`_ directory.

Overview of tools/llm Directory
-------------------------------

The ``tools/llm`` directory provides the following tools to compile LLM models from Huggingface:

* **run_llm.py**: Main entry point for model compilation, generating outputs, and benchmarking
* **Static Cache Utilities**: ``static_cache_v1.py`` and ``static_cache_v2.py`` for KV cache optimization
* **SDPA Attention**: ``sdpa_converter.py`` and ``register_sdpa.py`` for registering scaled dot-product attention converter and lowering pass.
* **Testing Components**: Model-specific test files for validation
* **Utility Functions**: ``utils.py`` and ``cache_utils.py`` for common operations

Supported Models
----------------
We have officially verified support for the following LLM families:

.. list-table::
   :widths: 20 40 20 20
   :header-rows: 1

   * - Model Series
     - HuggingFace Model Card
     - Precision
     - KV Cache Support ?
   * - GPT-2
     - gpt2
     - FP16, FP32
     - Yes
   * - LLaMA 2
     - meta-llama/Llama-2-7b-chat-hf
     - FP16, FP32
     - Yes
   * - LLaMA 3.1
     - meta-llama/Llama-3.1-8B-Instruct
     - FP16, FP32
     - Yes
   * - LLaMA 3.2
     - | meta-llama/Llama-3.2-1B-Instruct
       | meta-llama/Llama-3.2-3B-Instruct
     - FP16, FP32
     - Yes
   * - Qwen 2.5
     - | Qwen/Qwen2.5-0.5B-Instruct
       | Qwen/Qwen2.5-1.5B-Instruct
       | Qwen/Qwen2.5-3B-Instruct
       | Qwen/Qwen2.5-7B-Instruct
     - FP16, FP32
     - Yes

Getting Started with run_llm.py
-------------------------------

The main entry point is ``run_llm.py``, which provides a complete workflow for model compilation and benchmarking.

Basic Usage
^^^^^^^^^^^

.. code-block:: bash

   python tools/llm/run_llm.py \
     --model meta-llama/Llama-3.2-1B-Instruct \
     --prompt "What is parallel programming?" \
     --precision FP16 \
     --num_tokens 128 \
     --cache static_v2 \
     --benchmark

Key Arguments
^^^^^^^^^^^^^

* ``--model``: Name or path of the HuggingFace LLM
* ``--tokenizer``: (Optional) Tokenizer name; defaults to model name
* ``--prompt``: Input prompt for text generation
* ``--precision``: Precision mode (``FP16``, ``FP32``)
* ``--num_tokens``: Number of output tokens to generate
* ``--cache``: KV cache type (``static_v1``, ``static_v2``, or empty for no KV caching)
* ``--benchmark``: Enable benchmarking mode for performance comparison
* ``--enable_pytorch_run``: Also run and compare PyTorch baseline


Other Usage Examples
^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash

   # Compare different models performance
   python tools/llm/run_llm.py --model gpt2 --benchmark --enable_pytorch_run
   python tools/llm/run_llm.py --model meta-llama/Llama-3.2-1B-Instruct --benchmark --enable_pytorch_run

   # Generate the outputs (disable benchmarking) by specifying the number of tokens to generate. Default = 128
   python tools/llm/run_llm.py --model gpt2 --prompt "What is parallel programming?" --num_tokens 128
   python tools/llm/run_llm.py --model meta-llama/Llama-3.2-1B-Instruct --prompt "What is parallel programming?" --num_tokens 128

   # Test different caching approaches
   python tools/llm/run_llm.py --model meta-llama/Llama-3.2-1B-Instruct --cache static_v1
   python tools/llm/run_llm.py --model meta-llama/Llama-3.2-1B-Instruct --cache static_v2

   # Compare FP16 vs FP32 performance
   python tools/llm/run_llm.py --model Qwen/Qwen2.5-1.5B-Instruct --precision FP16 --benchmark
   python tools/llm/run_llm.py --model Qwen/Qwen2.5-1.5B-Instruct --precision FP32 --benchmark


KV Caching in Torch-TensorRT
---------------------------------

We provide two versions of static KV caching: `static_cache_v1 <https://github.com/pytorch/TensorRT/blob/main/tools/llm/static_cache_v1.py>`_ and `static_cache_v2 <https://github.com/pytorch/TensorRT/blob/main/tools/llm/static_cache_v2.py>`_.
In both implementations, we add static KV cache tensors as model inputs/outputs without storing them as external memory.
The length of KV cache = input sequence length + output sequence length (specified by ``--num_tokens``). The number of heads and head dimension are determined by the model config.

Static Cache v1
^^^^^^^^^^^^^^^^

The ``static_cache_v1.py`` implements KV cache  in the model graph as follows: 

.. code-block:: python

    class StaticCacheV1Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, q, k, v, key_cache, value_cache, start_idx, end_idx, is_causal=True):
            # Concatenate new key/value pairs with existing cache
            new_key_cache = torch.cat((key_cache[:, :, :start_idx, :], k, key_cache[:, :, end_idx:, :]), dim=2)
            new_value_cache = torch.cat((value_cache[:, :, :start_idx, :], v, value_cache[:, :, end_idx:, :]), dim=2)
            
            # Compute attention using the updated cache
            attn_output = torch._C._nn.scaled_dot_product_attention(
                q, 
                new_key_cache[:, :, :end_idx, :], 
                new_value_cache[:, :, :end_idx, :], 
                dropout_p=0.0, 
                is_causal=is_causal
            )

            return attn_output, new_key_cache, new_value_cache

In the above code, we concatenate the new key/value pairs with the existing cache and update it. To compute the attention, we use the updated cache and gather the corresponding keys/values from the cache up until and including the current token index.
The above code is actually implemented as a FX graph transformation pass. We register it as a Torch-TensorRT lowering pass using the decorator ``@_aten_lowering_pass`` when we import the ``static_cache_v1.py`` module.

.. note::
   The ``start_idx`` and ``end_idx`` are the start and end indices of the current token in the cache. For prefill phase, ``start_idx`` is 0 and ``end_idx`` is the input sequence length. 
   For decode phase, ``start_idx`` begins at the input sequence length and ``end_idx`` equals ``start_idx + 1``. The ``start_idx`` is incremented by 1 until the end of the sequence or we reach the maximum number of tokens to generate.


Static Cache v2
^^^^^^^^^^^^^^^^

The ``static_cache_v2.py`` is similar to ``static_cache_v1.py`` but it uses less number of slice operations. It implements KV cache in the model graph as follows: 

.. code-block:: python

    class StaticCacheV2Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, q, k, v, key_cache, value_cache, start_idx, end_idx, is_causal=True):
            concat_keys = torch.cat((key_cache[:, :, :start_idx, :], k), dim=2) 
            concat_values = torch.cat((value_cache[:, :, :start_idx, :], v), dim=2)
            new_key_cache = torch.cat((concat_keys, key_cache[:, :, end_idx:, :]), dim=2)
            new_value_cache = torch.cat((concat_values, value_cache[:, :, end_idx:, :]), dim=2)
            attn_output = torch._C._nn.scaled_dot_product_attention(
                  q, concat_keys, concat_values, dropout_p=0.0, is_causal=is_causal
            )

            return attn_output, new_key_cache, new_value_cache

In the above code, we concatenate the existing key/value cache with current key/value of the token. We use this to directly compute the attention and update the key/value cache inserting the current key/value.
The above code is actually implemented as a FX graph transformation pass. We register it as a Torch-TensorRT lowering pass using the decorator ``@_aten_lowering_pass`` when we import the ``static_cache_v1.py`` module.
The definitons of ``start_idx`` and ``end_idx`` are the same as ``static_cache_v1.py``.

After the model is compiled with static KV cache, the input signature of the model is changed. The new input signature is ``(input_ids, position_ids, key_cache_0, value_cache_0, ..., start_idx, end_idx)``. 
The number of key/value cache tensors is equal to the number of attention heads in the model. We can use the ``generate_with_static_cache`` function to generate the outputs.

Generating Outputs
------------------- 
We use custom `generate <https://github.com/pytorch/TensorRT/blob/main/tools/llm/utils.py#L112>`_ function to generate the outputs. This function performs standard autoregressive decoding without KV caching.
There is also a `generate_with_static_cache <https://github.com/pytorch/TensorRT/blob/main/tools/llm/utils.py#L141>`_ function that performs autoregressive decoding with KV caching.

The ``generate_with_static_cache`` function takes care of preparing the inputs to the model compiled with static KV cache.
The model inputs are ``input_ids``, ``position_ids``, ``key_cache_0``, ``value_cache_0``, ...., ``start_idx``, ``end_idx``.
We initialize the key/value cache tensors with zeros and for every token generated, the new key/value cache tensors are the outputs of the model.

SDPA Converter (sdpa_converter.py)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Converts scaled dot-product attention operation using TRT Python API.
* Supports causal and standard self-attention.

SDPA Registration (register_sdpa.py)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* This is a Torch-TensorRT lowering pass that replaces variants of SDPA with ``torch.nn.functional.scaled_dot_product_attention``.
* Registers the SDPA converter which is used for converting ``torch.nn.functional.scaled_dot_product_attention`` operation.


Limitations and Known Issues
----------------------------

* Sliding window attention (used in Gemma3 and Qwen 3 models) is not yet supported
* Some model architectures (e.g. Phi-4) have issues with exporting the torch model.

Requirements
^^^^^^^^^^^^

* Torch-TensorRT 2.8.0 or later
* Transformers v4.52.3