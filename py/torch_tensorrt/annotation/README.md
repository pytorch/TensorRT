# torch_tensorrt.annotation — custom_plugin descriptors

`torch_tensorrt.annotation` (aliased as `tta`) provides descriptor types and
factory functions for defining custom TensorRT AOT QDP plugins backed by
Triton, CuTile, or CuTeDSL kernels.

```python
import torch_tensorrt.annotation as tta
```

This module is **descriptor-only**: it builds spec objects that describe how a
plugin should be compiled and registered.  It does not patch `torch.export`,
add compilation hooks, or modify any torch-trt core path.

---

## Table of contents

1. [Quick start](#1-quick-start)
2. [Factory functions](#2-factory-functions)
3. [Spec types](#3-spec-types)
4. [QDP plugin flow](#4-qdp-plugin-flow)
5. [Running tests](#5-running-tests)

---

## 1. Quick start

```python
import torch_tensorrt.annotation as tta

# Triton AOT plugin
spec = tta.custom_plugin(tta.triton(my_launch_fn, configs=[{"BLOCK_SIZE": 128}]))

# CuTile plugin (Blackwell sm_100+)
spec = tta.custom_plugin(tta.cutile(my_cutile_kernel, arch=120))

# CuTeDSL plugin
spec = tta.custom_plugin(tta.cutedsl(my_cutedsl_kernel))
```

---

## 2. Factory functions

### `tta.triton(launch_fn, configs=None)`

Wraps a Triton kernel launch function.

- **`launch_fn`** — callable that launches the Triton kernel;
  signature `(input0, ..., output, stream, **config)`.
- **`configs`** — list of `dict` tactic configs; each becomes a separate
  QDP tactic.  Pass `None` for a single default tactic.

```python
@triton.jit
def _add_relu_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    tl.store(out_ptr + i, tl.maximum(0, tl.load(x_ptr+i, mask=mask) + tl.load(y_ptr+i, mask=mask)), mask=mask)

def launch_add_relu(x, y, out, stream, BLOCK=256):
    _add_relu_kernel[(triton.cdiv(x.numel(), BLOCK),)](x, y, out, x.numel(), BLOCK=BLOCK)

spec = tta.custom_plugin(tta.triton(launch_add_relu, configs=[{"BLOCK": 128}, {"BLOCK": 256}]))
```

### `tta.cutile(launch_fn, arch=None, configs=None)`

Wraps a CuTile (cuda-tile) kernel.  Requires Blackwell (sm_100+) and the
`cuda-tile` package.

- **`arch`** — SM architecture integer (e.g. `120` for sm_120).
- **`configs`** — list of tactic dicts.

```python
spec = tta.custom_plugin(tta.cutile(my_cutile_fn, arch=120, configs=[{"TILE_M": 64}]))
```

### `tta.cutedsl(launch_fn, configs=None)`

Wraps a CuTeDSL kernel (`nvidia-cutlass-dsl`).

```python
spec = tta.custom_plugin(tta.cutedsl(my_cutedsl_fn))
```

### `tta.custom_plugin(impl)`

Builds a `CustomPluginSpec` from a kernel spec (`TritonSpec`, `CuTileSpec`, or
`CuTeDSLSpec`).  Computes a deterministic QDP `op_name` from the kernel
function identity and config hash.

```python
spec = tta.custom_plugin(tta.triton(launch_fn, configs=[{"BLOCK": 256}]))
# spec.op_name  — deterministic string like "tta::launch_fn_a3f2c1"
```

---

## 3. Spec types

All spec types are plain frozen dataclasses — they carry no mutable state and
are safe to hash, compare, and cache.

| Type | Returned by | Description |
|------|-------------|-------------|
| `CustomPluginSpec` | `custom_plugin()` | AOT QDP plugin descriptor; holds `impl` (`TritonSpec` \| `CuTileSpec` \| `CuTeDSLSpec`) and computed `op_name` |
| `TritonSpec` | `triton()` | Triton kernel launch function + tactic configs |
| `CuTileSpec` | `cutile()` | CuTile kernel + target arch + tactic configs |
| `CuTeDSLSpec` | `cutedsl()` | CuTeDSL kernel + tactic configs |

---

## 4. QDP plugin flow

`tta.custom_plugin` produces a descriptor.  When you call
`register_custom_plugin(spec, inputs)` (from `_custom_plugin._descriptor`) the
module:

1. Derives a deterministic `op_name` from the kernel function + config hash.
2. Registers `@trtp.register("tta::op_name")` — the shape/dtype descriptor
   function derived symbolically from the kernel signature.
3. Registers `@trtp.aot_impl("tta::op_name")` — the AOT implementation
   function that returns `(kernel_name, ptx_or_cubin, KernelLaunchParams,
   SymIntExprs)`.
4. Uses a process-level lock + double-checked locking to prevent duplicate
   registration in multi-threaded pytest-xdist workers.

The QDP AOT path means **no Python is needed at TRT engine runtime** — the
compiled kernel is embedded directly.

```
tta.triton(launch_fn, configs)
    └─► TritonSpec
            └─► tta.custom_plugin(spec)
                    └─► CustomPluginSpec(op_name, impl)
                            └─► register_custom_plugin(spec, inputs)
                                    ├─► @trtp.register  (shape descriptor)
                                    └─► @trtp.aot_impl  (PTX/cubin → TRT)
```

---

## 5. Running tests

Unit tests are CPU-only (no GPU required) and live in
`tests/py/annotation/unit/`.

```bash
# From inside the dev Docker container:
python -m pytest tests/py/annotation/unit/ -n 4 --tb=short -v
```

Test files:

| File | What it covers |
|------|---------------|
| `test_specs.py` | `TritonSpec`, `CuTileSpec`, `CuTeDSLSpec` construction and hashing |
| `test_specs_custom_plugin.py` | `CustomPluginSpec` and `custom_plugin()` factory |
| `test_signature_binder.py` | TRT signature derivation and binding |
| `test_layer_metadata.py` | `AnnotationMetadata` encode/decode round-trip |
| `test_plugin_lowering.py` | QDP plugin lowering path |
