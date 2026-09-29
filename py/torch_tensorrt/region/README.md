# Explicit PyTorch regions (experimental MVP)

Use a scope to keep selected tensor operations together in PyTorch. Supported
operations outside the scope remain eligible for TensorRT.

```python
import torch
import torch_tensorrt


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = torch.nn.Linear(16, 16)
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(16, 16), torch.nn.Softmax(dim=-1)
        )
        self.mlp = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU())

    def forward(self, x):
        x = self.pre(x)
        with torch_tensorrt.region.execute_in_torch(name="attention"):
            x = self.attention(x)
        return self.mlp(x)


model = Model().eval().cuda()
x = torch.randn(4, 16, device="cuda")
compiled = torch_tensorrt.compile(
    model,
    inputs=[x],
    ir="dynamo",
    strict=True,
    offload_module_to_cpu=False,
    min_block_size=1,
)
with torch.no_grad():
    torch.testing.assert_close(compiled(x), model(x), atol=1e-4, rtol=1e-4)
```

`name` is optional and only labels diagnostics. Two scopes with the same name
are separate occurrences. Only the marked invocation is affected, not every call
to the same module. “In PyTorch” does not mean “on CPU”: CUDA tensors stay on CUDA.
Surrounding TensorRT placement still depends on converter support and
`min_block_size`; the example lowers that threshold for small demonstration graphs.

## How the boundary survives compilation

1. The public compile entry opens a private capture session. Strict Dynamo export
   records paired markers and tags each captured operation in the scope.
2. Before lowering, the compiler verifies both the pairing and the tags, finds
   values entering/leaving the scope, and extracts one child FX graph. Parameters
   and buffers used by that child become explicit inputs.
3. A temporary higher-order operator preserves the child through decomposition.
   It is then replaced by one opaque PyTorch `call_module` node.
4. Both partitioners treat that call as unsupported for TensorRT. The surrounding
   graph can become engines; the child executes its captured PyTorch operations.
   Audits check that rewriting, partitioning, and engine construction did not
   lose the child or put it inside an accelerated partition.

```text
TensorRT pre-processing -> PyTorch region child -> TensorRT post-processing
```

The internal manifest retains the occurrence ID, reference child graph, boundary
names, and PyTorch owner. It is a starting point for later region inspection and
kernel generation, not yet a public provider API. No marker executes at runtime.

## Initial limits

- Use the direct `torch_tensorrt.compile(model, ir="dynamo", ...)` entry above.
  Annotations are no-ops in eager execution, standalone `torch.export`, and other
  compiler sessions. A separately exported model has already lost the annotation.
- Eval/inference, static shapes, functional bodies, Tensor boundary values,
  read-only state, and one device per region. Multiple Tensor outputs and
  noncontiguous inputs are supported.
- Nested scopes, graph breaks, control flow inside the region, mutation, random
  operations, and outputs aliasing region inputs are rejected. Variable
  reassignment (`x = ...`) is fine. This is not an escape hatch for Python code
  that Dynamo cannot export.
- Disable CPU weight offload, full compilation, autocast, resource partitioning,
  refit, and weight streaming. Cross-compilation and engine-only output are not
  supported. Conflicting dry runs return a reference graph and report conflicts
  without building engines or offloading state.
- Empty/dead pure scopes are no-ops. Pattern matching, custom-kernel replacement,
  intermediate sample collection, and code generation are not implemented yet.

Capture currently relies on private PyTorch ordered-effect/annotation APIs and
`hints_wrapper`. Initial qualification uses PyTorch
`2.14.0.dev20260713+cu130`; this is not a compatibility claim for older releases.
Missing capture capabilities produce an explicit error when an annotation is used.

## Save and inspect

The supported save path preserves the already chosen placement without retracing:

```python
torch_tensorrt.save(
    compiled,
    "model.ep",
    output_format="exported_program",
    retrace=False,
    use_legacy_exporter=True,
)
restored = torch_tensorrt.load("model.ep").module()
```

Serialization inlines the PyTorch child on a copy of the graph and keeps primitive
region provenance. Reloading executes the already partitioned model; it does not
recreate a user annotation for a new optimization pass.

For compiler development, inspect `compiled.graph` and the internal
`torch_tensorrt.dynamo.regions.get_region_records(compiled)` manifest. Run the
focused tests with:

```bash
python -m pytest -n 0 tests/py/dynamo/regions
```
