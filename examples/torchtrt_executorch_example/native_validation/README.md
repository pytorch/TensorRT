# Native model validation and process-memory measurements

These applications load an existing ExecuTorch program through its C++ `Module`
API. `validate_pte` compares every output against offline PyTorch references;
`benchmark_pte` measures a fixed input snapshot. Neither invokes Python or
LibTorch. They reuse the TensorRT and CUDA/AOTInductor backend targets from
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
cmake --build "$build" --target validate_pte benchmark_pte -j2
cmake --install "$build" --prefix "$deployment" --component native_validation
```

The install directory contains `bin/` and any shared CUDA support libraries in
`lib/`. TensorRT and CUDA SDK/driver libraries remain target dependencies. Inspect
`ldd` on the target; do not copy a development machine's driver into deployment.
The install component stages only these applications and their shared CUDA
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

## Compare execution modes

Use the same checkpoint, inputs, precision and serialized artifact. Do not
compile during measurement. `benchmark_python.py` accepts `--mode eager` or
`--mode executorch`, and `--model-family bevformer` or `segformer`. Supply the
appropriate example directory as `--source-dir` and the checkpoint path.
BEVFormer's mixed artifact also requires `--data`.

```bash
python "$validation/benchmark_python.py" \
  --mode executorch --model-family segformer \
  --source-dir "$segformer_source" --checkpoint "$checkpoint" \
  --model "$bundle/model.pte" --case "$one_fixture" \
  --output "$python_results" --warmup 20 --iterations 100

# Native comparison: run on the host, outside Docker.
bash "$validation/benchmark_host.sh" \
  "$deployment/bin/benchmark_pte" "$bundle" "$one_fixture" \
  "$native_results" "$deployment/lib" "$cuda_root" 20 100 3
python "$validation/summarize_benchmark.py" "$native_results" \
  --output "$native_summary"
```

Repeat Python runs in fresh processes as well; do not compare one Python
process with several native processes as if those were equivalent samples.
Each process performs one first call, warmups and measured calls, with full
output checks before warmup, after warmup and after measurement. Timing starts
with prepared CPU tensors and ends with completed inference and owned CPU
outputs. It excludes image decoding, preprocessing and reference comparison.

`process_memory.py` and the native benchmark record Linux RSS, PSS and lifetime
peak RSS in KiB (1024 bytes); divide by 1024 for MiB. These include runtime
libraries, fixtures, reference tensors and application buffers. They are not
total GPU memory and must not be summed with CUDA free-memory snapshots. Peak
RSS is a historical maximum, not an additional allocation. The Python control
imports Torch and uses Torch tensors; it does not test a Torch-free Python API.

The October 2026 DRIVE measurements used Python inside the authoring container
and native C++ on the host. This is a deployment-configuration comparison,
not an isolated measurement of LibTorch's size. This integration has not yet
repeated those hardware measurements on the rebased source.

## Helper tests

```bash
python -m unittest discover -s "$validation" -p 'test_*.py' -v
```

These dependency-light tests exercise report validation, memory parsing and
model profiles. They do not execute a GPU model.
