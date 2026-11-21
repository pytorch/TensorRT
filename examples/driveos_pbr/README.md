# DRIVE Thor U benchmark handoff

This package benchmarks BEVFormer-Tiny and SegFormer-B0 on a **14-SM DRIVE
Thor U running Linux**. It compares three complete execution paths:

| Mode | Application | Model execution |
|---|---|---|
| `eager` | Python in the authoring container | Original FP32 PyTorch model, without `torch.compile` |
| `onnx` | Native C++ on DRIVE | ONNX-built TensorRT engine; BEVFormer includes a custom CUDA geometry plugin |
| `executorch` | Native C++ on DRIVE | ExecuTorch program containing Torch-TensorRT engines; BEVFormer also uses AOTInductor geometry |

The two compiled paths use different engines. This is a deployment comparison,
not a measurement of isolated runtime overhead. All paths use the same saved
inputs and expected FP32 outputs. Results may favor either implementation.

Start with [PBR.md](PBR.md) for the request text and [VALIDATION.md](VALIDATION.md)
for the handoff's verification results.

## Relationship to the blog figures

This handoff reproduces the measurements behind the native deployment comparison:

| Blog comparison | Handoff output |
|---|---|
| Inference latency for both models | `latency_ms` for `eager`, `onnx`, and `executorch` |
| BEVFormer-Tiny loading peak and retained process memory | `loading_peak_rss_mib` and `retained_rss_mib` |
| SegFormer-B0 CUDA free-memory decrease | `cuda_free_decrease_mib` |

It also records PSS and lifetime peak RSS for both models. It does not include
the separate CUDA Graph, green-context, shared-scratch, KV-cache, export-memory,
or camera-visualization experiments. The runner produces CSV/JSON data, not the
blog's styled plots.

Use results from one complete cohort when plotting. The handoff's cleaned-up
TensorRT runner releases engine-file bytes earlier than the older blog runner,
so its BEVFormer loading peak is different. See [VALIDATION.md](VALIDATION.md);
the older graph values are not exact reproduction targets for this package.

## Required environment and artifacts

| Component | Tested configuration |
|---|---|
| Target | DRIVE Thor U, 14 SMs, compute capability 11.0; Linux ARM64 |
| DRIVE OS | 7.2.5.0-46229577 |
| CUDA toolkit on target | 13.3 (`nvcc` 13.3.33) |
| TensorRT native library | 11.0.1.7; SONAME `libnvinfer.so.11` |
| Authoring container | `nvcr.io/nvidia/pytorch:26.06-py3` |
| Python / PyTorch | 3.12 / `2.16.0.dev20261009+cu132` |
| TorchVision / Transformers | `0.30.0.dev20261007+cu132` / 4.55.4 |
| ExecuTorch Python export package | `1.6.0.dev20261009+cu132` |
| Authoring TensorRT Python metadata | 11.0.0.114; this differs from the loaded native library |

This handoff does not target QNX, Orin, or a different TensorRT major version.
TensorRT plans and AOTInductor binaries are prepared artifacts for the tested
stack. The source-build path below rebuilds the native ExecuTorch runtime, TensorRT
delegate and both benchmark applications at pinned revisions. The authoring step
also builds the Torch-TensorRT Python compiler from source
and exports fresh ExecuTorch programs from the supplied checkpoints. The ONNX
TensorRT engines remain supplied artifacts.

Obtain `driveos-pbr-thoru-compact-20261120.tar.gz` and its SHA-256 file from the PBR
attachments. They contain third-party license notices and `bundle/`, including:

- `bevformer/`, `segformer/`: ONNX-built engines, checkpoints, and two fixtures each.
- `authoring/`: small Python dependency wheels, the matching TorchVision and
  ExecuTorch export wheels, and supplemental NGC libraries.
- `provenance/` and `manifest.json`: artifact origins, CPU-IO metadata changes,
  exact file sizes, and SHA-256 checksums.

The compressed archive is under 1 GB. Setup downloads the pinned PyTorch and
CUDA Python packages; Internet access and additional installation space are required.
Native binaries and ExecuTorch programs are built by the steps below.

Large binaries and model weights are **not in Git**. The source branch alone is
not a complete benchmark installation. Share the archive through the team's
artifact channel; do not put credentials or machine addresses in the request.

## 1. Arrange the handoff directory on DRIVE

Use a filesystem with at least 20 GiB free for extraction, the Python environment,
source dependencies and build outputs. Run the following in the directory containing the downloaded
archive and checksum:

```bash
sha256sum -c driveos-pbr-thoru-compact-20261120.tar.gz.sha256
export PBR_ROOT="$HOME/driveos-pbr"
mkdir -p "$PBR_ROOT"
tar -xzf driveos-pbr-thoru-compact-20261120.tar.gz -C "$PBR_ROOT"

```

The source, bundle, and output directories must all be under `PBR_ROOT` so the
eager process can access the same files through the container mount. Record the
Git commit used with the PBR.

## 2. Prepare the Python environment

Use a configured DRIVE authoring container with NVIDIA device access and mount
`PBR_ROOT` at `/pbr`. On a host where the NVIDIA Docker runtime is registered:

```bash
sudo docker run -d --name drive-pbr --runtime=nvidia \
  --device=/dev/host1x-fence --device=/dev/dri/renderD128 \
  --cap-add=SYS_NICE --ulimit memlock=-1 --ulimit stack=67108864 \
  --ipc=host --network=host -v "$PBR_ROOT:/pbr" \
  nvcr.io/nvidia/pytorch:26.06-py3 sleep infinity
sudo docker exec drive-pbr git clone --single-branch \
  --branch abose/PBR-file-driveOS \
  https://github.com/pytorch/TensorRT.git /pbr/TensorRT
sudo docker exec drive-pbr cp -a /pbr/TensorRT/examples/driveos_pbr /pbr/source
python3 "$PBR_ROOT/source/run.py" --bundle "$PBR_ROOT/bundle" --verify-only
sudo docker exec -it drive-pbr bash
```

The checkout runs inside the container; Git is not required on the target host.
Record its revision with `sudo docker exec drive-pbr git -C /pbr/TensorRT rev-parse HEAD`.
Inside that container:

```bash
bash /pbr/source/setup_python.sh /pbr
exit
```

`setup_python.sh` stores the Python environment in `/pbr/venv` and links
`/opt/pbr-venv` to it, keeping these files off Docker’s writable layer.
The matching TorchVision wheel avoids importing the incompatible stock container
version. The downloaded CUDA Python packages match the tested environment and
are separate from the target's CUDA 13.3 SDK. The bundle also retains supplemental
`libcufile` and `libcusparseLt` libraries from NGC, which DRIVE container mounts
can hide. NVSHMEM comes from the pinned Python package. This setup does not
replace host CUDA or TensorRT libraries.

An existing container is also usable: mount the handoff directory and pass its
actual mount point as `--container-root`. The runner uses `/opt/pbr-venv/bin/python`
by default; `--python` selects another matching interpreter.

## 3. Build the native runtime and applications from source

Inside the configured ARM64 container, after preparing `/opt/pbr-venv`:

```bash
bash /pbr/source/build_native.sh /pbr
```

This checks out two source dependencies at fixed commits under `/pbr/deps`:

- Torch-TensorRT native backend: `202fba495341f83c4d25b6b2dece95e0242c5369`.
- ExecuTorch: `a9665e25097b324a7116efa2bd6602287c9c064f`.

These are dependencies of the single PBR branch, not additional benchmark
branches to maintain. They keep the compiler and native runtime matched to the
tested export path. Using the latest upstream backend is a different
compatibility experiment. The source checkouts must be clean.

The script runs CMake configure, build and install. CMake compiles the
ExecuTorch runtime, its CUDA backend, the Torch-TensorRT delegate and both C++
applications. It installs executables in `/pbr/native/bin` and the newly built
shared libraries in `/pbr/native/lib`. No separate Bazel invocation is needed.
The PBR branch's current `.bazelrc` does not define `--config=driveos`.

This requires Git/network access for the pinned sources, CMake >=3.24, a C++
compiler, the DRIVE CUDA/TensorRT development files, and the prepared Python
build environment. `CUDAToolkit_ROOT` overrides the script's CUDA 13.3 default.
The script uses two build jobs to limit build-time memory pressure.

## 4. Build the Python compiler and export the programs

Run inside the same container, after the native source build:

```bash
bash /pbr/source/build_authoring.sh /pbr
export LD_LIBRARY_PATH=/pbr/bundle/authoring/lib
mkdir -p /pbr/cache/{tmp,triton,inductor}
export TMPDIR=/pbr/cache/tmp
export TRITON_CACHE_DIR=/pbr/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/pbr/cache/inductor
export XDG_CACHE_HOME=/pbr/cache
/opt/pbr-venv/bin/python /pbr/source/export_pte.py \
  --bundle /pbr/bundle --model segformer --output /pbr/exported/segformer
/opt/pbr-venv/bin/python /pbr/source/export_pte.py \
  --bundle /pbr/bundle --model bevformer --output /pbr/exported/bevformer
```

`build_authoring.sh` builds and installs Torch-TensorRT's Python compiler from
`deps/TensorRT-runtime`, the same pinned source used for the native delegate.
`PYTHON_ONLY=1` selects the Python authoring package; CMake already built the
native delegate. Setup downloads the matching Torch wheel; the ExecuTorch Python
wheel is supplied in the artifact archive. This does not build PyTorch itself from source.

The exporter loads the checkpoint, captures the PyTorch graph, compiles supported
regions with TensorRT, and saves the ExecuTorch program. BEVFormer geometry is
compiled through ExecuTorch's CUDA/AOTInductor backend. Outputs are:

- `exported/segformer/model.pte`: TensorRT-backed SegFormer.
- `exported/bevformer/model.pte` and `aoti_cuda_blob.ptd`: mixed BEVFormer program.
- Per-model graph listings and `export-report.json`: versions, artifact hashes,
  and eager/compiled comparisons against both saved input snapshots.

The export uses FP32 with TF32 disabled and configures caller-owned CPU inputs
and outputs through `MemoryPlanningPass`, before serialization. It does not
modify the serialized program afterward. Each output directory must be new.
An `EXPORT_PASS` confirms export checks; the next step checks the saved program
in the native runtime. Fresh engine builds can choose different tactics, so
report their new measurements as a new cohort rather than expecting identical
historical graph values.

## 5. Run the benchmark on the target host

Keep other GPU workloads stopped for the comparison. Record the board's power
mode and clock policy; the runner does not change clocks, drop caches, or reboot.
Run the native applications outside the authoring container:

```bash
export LD_LIBRARY_PATH="/usr/local/cuda-13.3/targets/aarch64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
sudo env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" python3 "$PBR_ROOT/source/run.py" \
  --bundle "$PBR_ROOT/bundle" --output "$PBR_ROOT/results" \
  --native-dir "$PBR_ROOT/native" --program-dir "$PBR_ROOT/exported" \
  --container drive-pbr --container-root /pbr
```

`sudo` allows this command to invoke Docker. It is unnecessary when the current
user already has Docker access. All native processes use the same user.
The `--program-dir` option selects the freshly exported programs; the input
fixtures, checkpoints, and ONNX engines still come from `bundle`. The runner
records the fresh program hashes without changing the original bundle.
The `--native-dir` option selects the rebuilt executables and shared libraries.
Both options are required for the full comparison. The configuration
record includes hashes of the selected rebuilt files.
The output directory must be new. A failed command or output mismatch stops the
matrix and leaves its log; it does not produce a successful summary.

For a native-only check, omit the container arguments and add:

```bash
--modes onnx executorch
```

For one model, add `--models bevformer` or `--models segformer`. Every selected
model/mode still runs six fresh processes. A single diagnostic invocation is:

```bash
LD_LIBRARY_PATH="$PBR_ROOT/native/lib:$LD_LIBRARY_PATH" \
  "$PBR_ROOT/native/bin/benchmark_et" \
  "$PBR_ROOT/exported/bevformer" "$PBR_ROOT/bundle/bevformer/case0" \
  "$PBR_ROOT/single-bevformer" bevformer
```

To run the ONNX engine, replace `benchmark_et` with `benchmark_trt` and use
`"$PBR_ROOT/bundle/bevformer"` as its first argument. The applications
set the same timing and correctness protocol; there are no hidden tuning flags.

## What is measured

- **36 processes total:** two models × three modes × two input snapshots × three
  launches. The six permutations of mode order balance which mode runs first.
- Each process runs one first call, 20 warmup calls, and 100 measured calls.
  Every output is checked after the first call, after warmup, and after timing
  using `abs(actual-reference) <= 3e-4 + 3e-4 * abs(reference)`.
- Latency starts with prepared CPU tensors and ends with completed inference and
  outputs in reusable CPU buffers. It includes CPU/GPU transfers and synchronization.
  Fixture reads, accuracy checks, preprocessing, and detection postprocessing are
  outside timing. The first call is recorded separately, not included in the median.
- Both native applications register and reuse CPU input/output buffers. The
  ExecuTorch programs expose caller-owned CPU IO; their internal GPU allocations
  and delegate blobs are retained. The exporter selects this IO planning explicitly; it is not the
  default export configuration. The eager control uses ordinary CPU tensors and
  reused GPU inputs. It is a development baseline, not an allocator ablation.
- CUDA Graph replay and shared activation scratch are not enabled by these runners.
  BEV fixtures repeat a fixed history snapshot; they are not a live driving sequence.

`summary.csv` reports the mean, minimum, and maximum of six process values. For
latency, each process value is the median of its 100 measured calls. Ranges are
process variation, not confidence intervals. For the relative blog plots, use
`mean eager latency / mean path latency` for speedup (eager = 1×), and
`100 * mean path memory / mean eager memory` for memory (eager = 100%). Compare
the same model and memory metric within the same cohort; keep raw ms/MiB data.

| Metric | Meaning |
|---|---|
| `loading_peak_rss_mib` | Process high-water RSS observed by the end of model/method loading, before fixtures |
| `retained_rss_mib` | Resident process pages after measured inference |
| `retained_pss_mib` | Resident pages with shared pages apportioned across processes, after measured inference |
| `lifetime_peak_rss_mib` | Process high-water RSS observed after inference; includes loading |
| `cuda_free_decrease_mib` | CUDA free memory after initialization minus free memory after inference |

RSS/PSS include fixture and reference buffers. CUDA free memory is a device-wide
reading affected by other CUDA allocations; it is neither process RSS nor total
board memory. Do not add these metrics together. Loading high-water RSS is a
historical maximum, not a separately allocated buffer.

Attach the whole results directory: it contains per-call timings, memory snapshots,
correctness reports, loaded-library maps, commands, and the aggregate summary.
Native processes must not load Python, LibTorch, or libc10; the runner checks this.
`device.txt` records the native GPU name, SM count, and CUDA versions.

## Optional: rebuild the geometry plugin

`geometry_plugin.cu` contains the fixed-shape BEV camera projection and temporal
rotation implementation used by the ONNX engine. It can be rebuilt separately:

```bash
mkdir -p /pbr/rebuilt
/usr/local/cuda-13.3/bin/nvcc -std=c++17 -O2 --fmad=false \
  -Xcompiler=-fPIC -shared -arch=sm_110 \
  -I/usr/include/aarch64-linux-gnu \
  -I/usr/local/cuda-13.3/thor/targets/aarch64-linux/include \
  /pbr/source/geometry_plugin.cu -lnvinfer -o /pbr/rebuilt/libbev_geometry.so
```

The bundle supplies the ONNX engines and fixed fixtures. The fresh `.pte` export
is a new build; retain its reports and benchmark results together.

## Files and provenance

| File | Purpose |
|---|---|
| `benchmark.h` | Shared fixture reader, pinned buffers, timing, correctness, and memory collection |
| `tensorrt.cpp` | Loads ONNX-built engines and runs `enqueueV3`; loads the BEV geometry plugin first |
| `executorch.cpp` | Loads the program, binds caller buffers and stream, and calls `Module::forward` |
| `setup_python.sh`, `authoring-requirements.txt` | Download and install pinned authoring dependencies |
| `build_authoring.sh`, `export_pte.py` | Build the Python compiler from source and export both models with caller-owned IO |
| `eager.py`, `models/` | Original FP32 inference baseline and model adapters |
| `run.py` | Verifies artifacts, runs fresh processes, and aggregates valid results |
| `test_run.py` | Tests artifact verification and rejection of incomplete measurements |
| `CMakeLists.txt`, `source_runtime.cmake`, `build_native.sh` | Build native runtime, delegate and two executables from pinned source |
| `PBR.md`, `VALIDATION.md`, `THIRD_PARTY.md` | Request, verification record, and source attribution |

Run the host-independent checks with:

```bash
python3 -m unittest discover -s "$PBR_ROOT/source" -p test_run.py -v
```

The pinned ExecuTorch exporter can leave the output node's tensor metadata
pointing to GPU intermediates after inserting device-to-host copies.
`CallerIOMemoryPlanning` refreshes that metadata before the normal planner runs.
The serialized-plan checks then require all public CPU IO to be caller-owned;
native output comparisons verify the resulting program. No installed ExecuTorch
source files or serialized `.pte` files are patched.
