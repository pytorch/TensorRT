Multi-device safe mode is a setting in Torch-TensorRT which allows the user to determine whether
the runtime checks for device consistency prior to every inference call.

There is a non-negligible, fixed cost per-inference call when multi-device safe mode, which is why
it is now disabled by default. It can be controlled via the following convenience function which
doubles as a context manager.
```python
# Enables Multi Device Safe Mode
torch_tensorrt.runtime.set_multi_device_safe_mode(True)

# Disables Multi Device Safe Mode [Default Behavior]
torch_tensorrt.runtime.set_multi_device_safe_mode(False)

# Enables Multi Device Safe Mode, then resets the safe mode to its prior setting
with torch_tensorrt.runtime.set_multi_device_safe_mode(True):
    ...
```
TensorRT requires that each engine be associated with the CUDA context in the active thread from which it is invoked.
Therefore, if the device were to change in the active thread, which may be the case when invoking
engines on multiple GPUs from the same Python process, safe mode will cause Torch-TensorRT to display
an alert and switch GPUs accordingly. If safe mode were not enabled, there could be a mismatch in the engine
device and CUDA context device, which could lead the program to crash.

One technique for managing multiple TRT engines on different GPUs while not sacrificing performance for
multi-device safe mode is to use Python threads. Each thread is responsible for all of the TRT engines
on a single GPU, and the default CUDA device on each thread corresponds to the GPU for which it is
responsible (can be set via `torch.cuda.set_device(...)`). In this way, multiple threads can be used in the same
Python scripts without needing to switch CUDA contexts and incur performance overhead by leveraging threads.
