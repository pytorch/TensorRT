# Torch-TensorRT parallelism for distributed inference

Examples in this folder demonstrates doing distributed inference on multiple devices with Torch-TensorRT backend.

1. Data parallel distributed inference based on [Accelerate](https://huggingface.co/docs/accelerate/usage_guides/distributed_inference)

Using Accelerate users can achieve data parallel distributed inference with Torch-TensorRt backend. In this case, the entire model
will be loaded onto each GPU and different chunks of batch input is processed on each device.

See the examples started with `data_parallel` for more details.

2. Tensor parallel distributed inference

Here we use torch.distributed as an example, but compilation with tensor parallelism is agnostic to the implementation framework as long as the module is properly sharded.

torchrun --nproc_per_node=2 tensor_parallel_llama2.py

3. Tensor parallel distributed inference using nccl ops plugin

 apt install libmpich-dev

 apt install libopenmpi-dev

 #For python3.10

 pip install tensorrt-llm

 For other python versions, you need to load the libnvinfer_plugin_tensorrt_llm.so. Please set that in the environment variable export trtllm_env={lib_path}. For example, we have already set the variable in initialize_distributed_env(). Note that won't work while running example, since it needs to be preset for the converter library to get.

 #then pip install the tensorrt and torch version compatible with installed torchTRT

 mpirun -n 2 --allow-run-as-root python tensor_parallel_simple_example.py

 #For other python

4. Tensor parallel distributed llama3 inference using nccl ops plugin

 apt install libmpich-dev

 apt install libopenmpi-dev

#For python3.10

 pip install tensorrt-llm

 For other python versions, you need to load the libnvinfer_plugin_tensorrt_llm.so

 #then pip install the tensorrt and torch version compatible with installed torchTRT

 mpirun -n 2 --allow-run-as-root python tensor_parallel_llama3.py
