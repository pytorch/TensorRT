# Native model validation

The `validate_pte` application loads an ExecuTorch program through its C++
`Module` API and compares every output against offline PyTorch references.
It invokes neither Python nor LibTorch. It reuses the TensorRT and CUDA/AOTInductor backend targets from
[`executorch_reference_runner`](../../executorch_reference_runner).

The examples support fixed-shape, contiguous FP32 tensors. The correctness
runner additionally implements BEVFormer's explicit recurrent-state protocol.
Image preprocessing, checkpoint loading and reference generation happen offline
in Python. No exported models, weights, images or benchmark results are shipped.

## Build and stage

In the configured authoring container, set paths for your source checkouts,
SDK and output directories. `CUDAToolkit_ROOT` must be the injected DRIVE CUDA
root, containing both the common and Thor header trees. For other platforms,
omit the DRIVE compatibility option and choose the appropriate GPU architecture.

```bash
repo=$(git rev-parse --show-toplevel)
validation="$repo/examples/torchtrt_executorch_example/native_validation"
cmake -S "$validation" -B "$build" \
  -DEXECUTORCH_SOURCE_DIR="$executorch_source" \
  -DTORCHTRT_ENABLE_DRIVE_OS_COMPAT=ON \
  -DCUDAToolkit_ROOT="$CUDA_HOME" \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=110 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$build" --target validate_pte -j2
cmake --install "$build" --prefix "$deployment" --component native_validation
```

The install directory contains `bin/` and any shared CUDA support libraries in
`lib/`. TensorRT and CUDA SDK/driver libraries remain target dependencies. Inspect
`ldd` on the target; do not copy a development machine's driver into deployment.
The install component stages only the application and its shared CUDA
support libraries, not the dependency projects' development files.

## Independent fixtures

Each fixture directory contains little-endian FP32 `input_N.bin` and
`output_N.bin`, with whitespace-separated dimensions in `*.shape`. Inputs and
outputs are numbered from zero. The native checker uses program metadata for
input sizes and the reference shape files for outputs. Missing or invalid
data, nonfinite values or numerical mismatches cause a nonzero exit status.

On the DRIVE host, use host paths for every argument:

```bash
bash "$validation/run_cases_host.sh" \
  "$deployment/bin/validate_pte" "$bundle" "$fixtures" \
  "$new_results" "$deployment/lib" "$cuda_root"
```

The wrapper supplies a clean loader environment, checks linked and live loaded
libraries for Python/LibTorch, preserves raw outputs and hashes, and refuses to
overwrite results. It includes `aoti_cuda_blob.ptd` automatically when present.
Keep the `.pte` and matching `.ptd` together. Artifact compatibility depends on
the exporter/runtime revisions and target SDK, not merely the filename.

## BEVFormer recurrence

The BEVFormer fixture generator writes a manifest containing lines such as:

```text
reset scene_00_frame_000
continue scene_00_frame_001
reset scene_01_frame_000
```

`--bev-sequence` loads the model once. Each reset clears previous BEV state;
each continuation uses the runner's own preceding output, not the fixture's
stored history tensor. Output storage is copied into application-owned memory
before the next call. This is a correctness loop with CPU boundaries, not a
zero-copy streaming implementation.

```bash
env -i PATH=/usr/bin:/bin \
  LD_LIBRARY_PATH="$deployment/lib:$cuda_root/targets/aarch64-linux/lib:$cuda_root/thor/targets/aarch64-linux/lib" \
  "$deployment/bin/validate_pte" \
  --model="$bundle/model.pte" --data="$bundle/aoti_cuda_blob.ptd" \
  --bev-sequence="$fixtures/sequence.txt" --output="$new_results"
```
