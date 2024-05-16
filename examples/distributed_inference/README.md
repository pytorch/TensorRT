# Torch-TensorRT parallelism for distributed inference

Examples in this folder demonstrates doing distributed inference on multiple devices with Torch-TensorRT backend.

1. Data parallel distributed inference based on [Acclerate](https://huggingface.co/docs/accelerate/usage_guides/distributed_inference)

Using Accelerate users can achieve data parallel distributed inference with Torch-TensorRt backend. In this case, the entire model
will be loaded onto each GPU and different chunks of batch input is processed on each device.

See the examples started with `data_parallel` for more details.

2. Tensor parallel distributed inference

In development.
