# Torch-TensorRT parallelism for distributed inference

Examples in this folder demonstrates doing distributed inference on multiple devices with Torch-TensorRT backend.

## Data parallel distributed inference based on [Accelerate](https://huggingface.co/docs/accelerate/usage_guides/distributed_inference)

Using Accelerate users can achieve data parallel distributed inference with Torch-TensorRt backend. In this case, the entire model
will be loaded onto each GPU and different chunks of batch input is processed on each device.

See the examples [data_parallel_gpt2.py](https://github.com/pytorch/TensorRT/blob/main/examples/distributed_inference/data_parallel_gpt2.py) and [data_parallel_stable_diffusion.py](https://github.com/pytorch/TensorRT/blob/main/examples/distributed_inference/data_parallel_stable_diffusion.py) for more details.

## Tensor parallel distributed inference

Here we use torch.distributed as an example, but compilation with tensor parallelism is agnostic to the implementation framework as long as the module is properly sharded.

torchrun --nproc_per_node=2 tensor_parallel_llama2.py

## Tensor parallel distributed inference on a simple model using nccl ops plugin

 
We use [torch.distributed](https://pytorch.org/docs/stable/distributed.html) package to add shard the model with Tensor parallelism. The distributed ops (`all_gather` and `all_reduce`) are then expressed as TensorRT-LLM plugins to avoid graph breaks during Torch-TensorRT compilation. The [converters for these operators](https://github.com/pytorch/TensorRT/blob/main/py/torch_tensorrt/dynamo/conversion/custom_ops_converters.py#L25-L55) are already available in Torch-TensorRT. The functional implementation of ops is imported from `tensorrt_llm` package (to be more specific, only `libnvinfer_plugin_tensorrt_llm.so` is required). So we have two options here 

### Option 1: Install TensorRT-LLM

Follow the instructions to [install TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/installation/linux.html)

If the default installation fails due to issues like library version mismatches or Python compatibility, it is recommended to use Option 2. After a successful installation, ensure you test by running `import torch_tensorrt` to confirm it works without errors. The import might fail if the `tensorrt_llm` installation overrides `torch_tensorrt` dependencies. Option 2 is particularly advisable if you prefer not to install `tensorrt_llm` and its associated dependencies.

### Option 2: Link the TensorRT-LLM directly.

 Another alternative is to load the `libnvinfer_plugin_tensorrt_llm.so` directly. You can do this by 
  * Downloading the [tensorrt_llm-0.16.0](https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-0.16.0-cp310-cp310-linux_x86_64.whl#sha256=f86c6b89647802f49b26b4f6e40824701da14c0f053dbda3e1e7a8709d6939c7) wheel file from the NVIDIA python index. 
  * Extract the wheel file to a directory and you can find the `libnvinfer_plugin_tensorrt_llm.so` library under `tensorrt_llm/libs` directory.
  * Please set the environment variable TRTLLM_PLUGINS_PATH to the above extracted path at the [initialize_distributed_env()](https://github.com/pytorch/TensorRT/blob/54e36dbafe567c75f36b3edb22d6f49d4278c12a/examples/distributed_inference/tensor_parallel_initialize_dist.py#L45) call.


After configuring the TensorRT-LLM or the TensorRT-LLM plugin library path, please run the following command which illustrates tensor parallelism of a simple model and compilation with Torch-TensorRT

```py
mpirun -n 2 --allow-run-as-root python tensor_parallel_simple_example.py
```

We also provide a tensor paralellism compilation example on a more advanced model like `Llama-3`. Here's the command to run it

```py
mpirun -n 2 --allow-run-as-root python tensor_parallel_llama3.py
```
