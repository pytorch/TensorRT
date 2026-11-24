# Native model validation

The `validate_pte` application loads an ExecuTorch program through its C++
`Module` API and compares every output against offline PyTorch references.
It invokes neither Python nor LibTorch. It reuses the TensorRT and CUDA/AOTInductor backend targets from
[`executorch_reference_runner`](../../executorch_reference_runner).

The examples support fixed-shape, contiguous FP32 tensors. The correctness
runner additionally implements BEVFormer's explicit recurrent-state protocol.
Image preprocessing, checkpoint loading and reference generation happen offline
in Python. No exported models, weights, images or benchmark results are shipped.

`benchmark_pte` measures inference latency and process memory for fixed inputs.

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

## Reuse application buffers for inference

The default benchmark materializes outputs into application-owned CPU buffers.
The optimized path exports public CPU inputs and outputs without arena storage,
binds reusable output buffers with `Module::set_output`, and optionally registers
input/output storage with CUDA. Registration happens once before warmup, outside
the timed calls. The model, precision, GPU computation and correctness tolerance
are unchanged.

The flags are deliberately separate to compare:

| Configuration | Export | Native benchmark |
|---|---|---|
| Original CPU IO | Default | Default |
| Caller-owned CPU IO | `--caller-owned-io` | `--caller-owned-io` |
| Registered caller-owned CPU IO | `--caller-owned-io` | `--caller-owned-io --register-host-memory` |

### Export

Run from this checkout in the configured authoring environment. Use new output
directories; the exporters refuse to overwrite an existing result.

```bash
python examples/torchtrt_executorch_example/bevformer/export_full.py \
  --checkpoint "$BEV_CHECKPOINT" --output-dir "$BEV_BUNDLE" --caller-owned-io

python examples/torchtrt_executorch_example/segformer/export_full.py \
  --checkpoint "$SEG_CHECKPOINT" --output "$SEG_BUNDLE" --caller-owned-io
```

The shared `caller_owned_io.py` selects `alloc_graph_input=False` and
`alloc_graph_output=False`. It refreshes output metadata after CUDA copies so
that the final CPU outputs, rather than GPU intermediates, are excluded from
memory planning. This workaround came from the pinned PBR exporter; it does not
patch an installed ExecuTorch package. Export fails if any serialized public
input or output still has an arena allocation. The existing full-output Python
checks remain enabled.

### Build and run

Build `benchmark_pte` using the instructions above. On DRIVE,
set `RUNNER` to that binary, `BUNDLE` to the freshly exported model directory,
`FIXTURE` to a saved input/reference directory and `RESULTS` to a new directory.
BEVFormer fixtures are prepared with `prepare_native_cases.py`; SegFormer's
exporter writes `seed_7`, `seed_19` and `seed_51` fixture directories.

```bash
data=()
if [[ -f "$BUNDLE/aoti_cuda_blob.ptd" ]]; then
  data=("--data=$BUNDLE/aoti_cuda_blob.ptd")
fi
"$RUNNER" --model="$BUNDLE/model.pte" "${data[@]}" \
  --case="$FIXTURE" --output="$RESULTS" \
  --warmup=20 --iterations=100 \
  --caller-owned-io --register-host-memory
```

The runner rejects arena-planned public tensors in caller-owned mode and checks
that each returned output still points to the bound application buffer. It
checks every value before and after timing. Registered memory is released only
after GPU work completes and before the buffers are destroyed. Overlapping host
pages are registered once; partial registration failures release earlier pages.
Unsupported CUDA host registration is an explicit error, not a silent fallback.

Repeat each configuration in fresh processes for the same two input snapshots,
three launches per snapshot. Use separate result directories and the same
clocks, precision and input/output boundary. Timings include model execution,
GPU completion and delivery into CPU result buffers. `metadata.tsv` records the
chosen options; `latency.tsv`, `memory.tsv` and `comparisons.tsv` retain the
measurements and correctness checks. Do not combine runs with different options
into a single mean.

### Validation scope

This integrates the caller-owned buffer strategy into the example runner. A
fresh DRIVE build, export and benchmark is required to measure its effect on
this branch. The baseline and optimized modes use the same model, precision and
correctness tolerances; no CUDA Graph or shared-scratch settings are changed.

CPU regression checks:

```bash
python -m unittest discover \
  -s examples/torchtrt_executorch_example/native_validation -p 'test_*.py' -v
```

The output-spec tests need PyTorch but no GPU. The host-registration tests compile
against a fake CUDA API and test page merging and cleanup, not GPU transfer speed.
