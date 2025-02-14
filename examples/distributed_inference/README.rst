Torch-TensorRT Parallelism for Distributed Inference
====================================================

Examples in this folder demonstrate distributed inference on multiple devices with the Torch-TensorRT backend.

Data Parallel Distributed Inference based on `Accelerate <https://huggingface.co/docs/accelerate/usage_guides/distributed_inference>`_
---------------------------------------------------------------------------------------------------------------

Using Accelerate, users can achieve data parallel distributed inference with the Torch-TensorRT backend.
In this case, the entire model will be loaded onto each GPU, and different chunks of batch input are processed on each device.

See the examples:

- `data_parallel_gpt2.py <https://github.com/pytorch/TensorRT/blob/main/examples/distributed_inference/data_parallel_gpt2.py>`_
- `data_parallel_stable_diffusion.py <https://github.com/pytorch/TensorRT/blob/main/examples/distributed_inference/data_parallel_stable_diffusion.py>`_

for more details.

Tensor Parallel Distributed Inference
--------------------------------------

Here, we use `torch.distributed` as an example, but compilation with tensor parallelism is agnostic to the implementation framework as long as the module is properly sharded.

.. code-block:: bash

    torchrun --nproc_per_node=2 tensor_parallel_llama2.py

Tensor Parallel Distributed Inference on a Simple Model using NCCL Ops Plugin
------------------------------------------------------------------------------

We use `torch.distributed <https://pytorch.org/docs/stable/distributed.html>`_ to shard the model with Tensor parallelism.
The distributed operations (`all_gather` and `all_reduce`) are then expressed as TensorRT-LLM plugins to avoid graph breaks during Torch-TensorRT compilation.
The `converters for these operators <https://github.com/pytorch/TensorRT/blob/main/py/torch_tensorrt/dynamo/conversion/custom_ops_converters.py#L25-L55>`_ are already available in Torch-TensorRT.
The functional implementation of ops is imported from the `tensorrt_llm` package (specifically, `libnvinfer_plugin_tensorrt_llm.so` is required).

We have two options:

Option 1: Install TensorRT-LLM
-------------------------------

Follow the instructions to `install TensorRT-LLM <https://nvidia.github.io/TensorRT-LLM/installation/linux.html>`_.

If the default installation fails due to issues like library version mismatches or Python compatibility, consider using Option 2.
After a successful installation, test by running:

.. code-block:: python

    import torch_tensorrt

to ensure it works without errors.
The import might fail if `tensorrt_llm` overrides `torch_tensorrt` dependencies.
Option 2 is preferable if you do not wish to install `tensorrt_llm` and its dependencies.

Option 2: Link the TensorRT-LLM Directly
-----------------------------------------

Alternatively, you can load `libnvinfer_plugin_tensorrt_llm.so` manually:

1. Download the `tensorrt_llm-0.16.0 <https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-0.16.0-cp310-cp310-linux_x86_64.whl#sha256=f86c6b89647802f49b26b4f6e40824701da14c0f053dbda3e1e7a8709d6939c7>`_ wheel file from NVIDIA's Python index.
2. Extract the wheel file to a directory and locate `libnvinfer_plugin_tensorrt_llm.so` under the `tensorrt_llm/libs` directory.
3. Set the environment variable `TRTLLM_PLUGINS_PATH` to the extracted path at the `initialize_distributed_env() <https://github.com/pytorch/TensorRT/blob/54e36dbafe567c75f36b3edb22d6f49d4278c12a/examples/distributed_inference/tensor_parallel_initialize_dist.py#L45>`_ call.

After configuring TensorRT-LLM or the TensorRT-LLM plugin library path, run the following command to illustrate tensor parallelism of a simple model and compilation with Torch-TensorRT:

.. code-block:: bash

    mpirun -n 2 --allow-run-as-root python tensor_parallel_simple_example.py

We also provide a tensor parallelism compilation example on a more advanced model like `Llama-3`. Run the following command:

.. code-block:: bash

    mpirun -n 2 --allow-run-as-root python tensor_parallel_llama3.py

Tutorials
-----------------------------------------
* :ref:`tensor_parallel_llama3`: Illustration of distributed inference on multiple devices with the Torch-TensorRT backend.