# Torch-TensorRT Python frontend

Torch-TensorRT compiles PyTorch models to TensorRT with `torch.export` or
`torch.compile`. TorchScript compilation and serialization have been removed.

```python
import torch
import torch_tensorrt

model = torch.nn.Linear(16, 8).eval().cuda()
inputs = [torch.randn(4, 16, device="cuda")]
compiled = torch_tensorrt.compile(model, ir="dynamo", inputs=inputs)
torch_tensorrt.save(compiled, "model.ep", inputs=inputs)
loaded = torch_tensorrt.load("model.ep").module()
outputs = loaded(*inputs)
```

For C++ deployment, save an AOTInductor package with
`output_format="aot_inductor"` or export an ExecuTorch program using
`torch_tensorrt.executorch.export`.
