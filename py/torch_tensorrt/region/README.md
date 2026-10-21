# Explicit regions (experimental)

See the [region tutorial](../../../docsrc/tutorials/execute_in_torch.rst)
for a complete example, supported settings, save/load instructions, and limits.

Use region annotations with direct Dynamo compilation, `strict=True`, and
`offload_module_to_cpu=False` on an eval-mode model with static input shapes.

- `execute_in_torch(name=None, backend="eager")` keeps the captured operations
  together in PyTorch. The optional `backend="inductor"` compiles that region on
  its first actual call, still outside TensorRT. These are the only accepted
  backends; the argument does not select a `torch.compile` tuning mode.
- `fuse_in_plugin(kernel=lambda: my_kernel, ...)` replaces the captured
  operations with a supplied Triton kernel or CUDA `KernelSpec` inside TensorRT.
  The compiler derives the operator signature and output metadata and registers
  the plugin and converter automatically. The zero-argument factory supplies
  the existing kernel outside tensor tracing; `name` is optional.

Plugin regions require static, contiguous CUDA tensor inputs and outputs.
The supplied kernel must compute the same values as the annotated PyTorch body;
registration checks tensor contracts, not numerical equivalence. The runnable
example includes a comparison against the original model.

The annotation is a no-op outside a Torch-TensorRT-owned compilation, including
ordinary eager execution and standalone `torch.export`. Setting
`backend="inductor"` does not compile a model called directly in eager mode.
The legacy exported-program save path supports `execute_in_torch` regions with
`backend="eager"` and plugin regions already built into TensorRT engines. Loading
a built plugin engine does not require registering the Python kernel again.
Plugin dry-run graphs cannot be saved because they have no built engines.
Saving an Inductor region raises an error because its compiled callable cannot
be preserved by the current exporter.

Run [execute_in_torch.py](../../../examples/dynamo/execute_in_torch.py) to run
three operations with PyTorch eager and Inductor, or
[fuse_in_plugin.py](../../../examples/dynamo/fuse_in_plugin.py) to
replace them with a custom kernel inside one TensorRT engine:

```bash
python examples/dynamo/execute_in_torch.py
python examples/dynamo/fuse_in_plugin.py
```

Start with [test_execute_in_torch.py](../../../tests/py/dynamo/regions/test_execute_in_torch.py)
for eager execution, capture, and compilation of a small model. Extended cases
live in `test_execute_in_torch_regressions.py`.

Run all region tests with:

```bash
python -m pytest -n 0 tests/py/dynamo/regions
```
