The code in this directory is similar to `torch_tensorrrt.fx`. We intend to make changes under `dynamo` namespace to ensure we
have the same top level API as `torch_tensorrt.ts.compile`. Right now, the usage is as follows

```
import torch_tensorrt
trt_module = torch_tensorrt.compile(
        module,
        ir="dynamo"
        torchtrt_inputs,
        enabled_precisions={torch.float32},
    )
```
This will internally call `torch_tensorrt.dynamo.compile` which has the same signature as `torch_tensorrt.ts.compile`. We intend to add features (existing in Torchscript backend for eg: torch_executed_ops, torch_executed_modules and many more) to this dynamo backend in the coming months.
