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
| Exported ExecuTorch revision | Development 1.6; use the supplied native SDK and program together |
| Authoring TensorRT Python metadata | 11.0.0.114; this differs from the loaded native library |

This handoff does not target QNX, Orin, or a different TensorRT major version.
TensorRT plans and AOTInductor binaries are prepared artifacts for the tested
stack. Rebuilding the applications is supported below; regenerating engines or
rebuilding the compiler/runtime stack is a separate experiment.

Obtain `driveos-pbr-thoru-20261117.tar.gz` and its SHA-256 file from the PBR
attachments. They contain third-party license notices and `bundle/`, including:

- `bevformer/`, `segformer/`: engines, programs, checkpoints, and two fixtures each.
- `bin/`: native benchmark applications; `lib/`: native CUDA backend shims.
- `sdk/`: frozen native headers and static libraries for rebuilding the applications.
- `authoring/`: pinned Torch wheel, Python dependency wheels, CUDA Python libraries,
  and supplemental NGC libraries needed by the eager baseline.
- `provenance/` and `manifest.json`: artifact origins, CPU-IO metadata changes,
  exact file sizes, and SHA-256 checksums.

Large binaries and model weights are **not in Git**. The source branch alone is
not a complete benchmark installation. Share the archive through the team's
artifact channel; do not put credentials or machine addresses in the request.

## 1. Arrange the handoff directory on DRIVE

Use a filesystem with at least 12 GiB free for extraction, the Python environment,
and build outputs. Run the following in the directory containing the downloaded
archive and checksum:

```bash
sha256sum -c driveos-pbr-thoru-20261117.tar.gz.sha256
export PBR_ROOT="$HOME/driveos-pbr"
mkdir -p "$PBR_ROOT"
tar -xzf driveos-pbr-thoru-20261117.tar.gz -C "$PBR_ROOT"

git clone --single-branch --branch abose/PBR-file-driveOS \
  https://github.com/pytorch/TensorRT.git "$PBR_ROOT/TensorRT"
cp -a "$PBR_ROOT/TensorRT/examples/driveos_pbr" "$PBR_ROOT/source"
python3 "$PBR_ROOT/source/run.py" --bundle "$PBR_ROOT/bundle" --verify-only
```

The source, bundle, and output directories must all be under `PBR_ROOT` so the
eager process can access the same files through the container mount. Record the
Git commit used with the PBR.

## 2. Prepare the Python baseline

Use a configured DRIVE authoring container with NVIDIA device access and mount
`PBR_ROOT` at `/pbr`. On a host where the NVIDIA Docker runtime is registered:

```bash
sudo docker run -d --name drive-pbr --runtime=nvidia \
  --ipc=host --network=host -v "$PBR_ROOT:/pbr" \
  nvcr.io/nvidia/pytorch:26.06-py3 sleep infinity
sudo docker exec -it drive-pbr bash
```

Inside that container:

```bash
python3 -m venv --system-site-packages /opt/pbr-venv
/opt/pbr-venv/bin/python -m pip install --no-deps \
  /pbr/bundle/authoring/wheels/torch-*.whl \
  /pbr/bundle/authoring/wheels/torchvision-*.whl
/opt/pbr-venv/bin/python -m pip install --no-index \
  --find-links=/pbr/bundle/authoring/wheels -r /pbr/source/requirements.txt
ln -s /pbr/bundle/authoring/nvidia /opt/pbr-venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="/pbr/bundle/authoring/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
/opt/pbr-venv/bin/python -c \
  'import torch; print(torch.__version__); print(torch.cuda.get_device_properties(0))'
exit
```

The matching TorchVision wheel avoids importing the incompatible stock container
version. The supplied CUDA Python libraries preserve the tested eager environment. They
are separate from the target's CUDA 13.3 SDK. Supplemental NGC libraries include
`libcufile`, `libcusparseLt`, and `libnvshmem_host`, which DRIVE container mounts
can hide. This setup does not replace host CUDA or TensorRT libraries.

An existing container is also usable: mount the handoff directory and pass its
actual mount point as `--container-root`. The runner uses `/opt/pbr-venv/bin/python`
by default; `--python` selects another matching interpreter.

## 3. Run the benchmark on the target host

Keep other GPU workloads stopped for the comparison. Record the board's power
mode and clock policy; the runner does not change clocks, drop caches, or reboot.
Run the native applications outside the authoring container:

```bash
export LD_LIBRARY_PATH="/usr/local/cuda-13.3/targets/aarch64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
sudo env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" python3 "$PBR_ROOT/source/run.py" \
  --bundle "$PBR_ROOT/bundle" --output "$PBR_ROOT/results" \
  --container drive-pbr --container-root /pbr
```

`sudo` allows this command to invoke Docker. It is unnecessary when the current
user already has Docker access. All native processes use the same user.
The output directory must be new. A failed command or output mismatch stops the
matrix and leaves its log; it does not produce a successful summary.

For a native-only check, omit the container arguments and add:

```bash
--modes onnx executorch
```

For one model, add `--models bevformer` or `--models segformer`. Every selected
model/mode still runs six fresh processes. A single diagnostic invocation is:

```bash
LD_LIBRARY_PATH="$PBR_ROOT/bundle/lib:$LD_LIBRARY_PATH" \
  "$PBR_ROOT/bundle/bin/benchmark_et" \
  "$PBR_ROOT/bundle/bevformer" "$PBR_ROOT/bundle/bevformer/case0" \
  "$PBR_ROOT/single-bevformer" bevformer
```

Replace `benchmark_et` with `benchmark_trt` to run the ONNX engine. The applications
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
  and delegate blobs are retained. This is an optimized prepared export, not the
  default export configuration. The eager control uses ordinary CPU tensors and
  reused GPU inputs. It is a development baseline, not an allocator ablation.
- CUDA Graph replay and shared activation scratch are not enabled by these runners.
  BEV fixtures repeat a fixed history snapshot; they are not a live driving sequence.

`summary.csv` reports the mean, minimum, and maximum of six process values. For
latency, each process value is the median of its 100 measured calls. Ranges are
process variation, not confidence intervals.

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

## Rebuild the native applications

Inside the configured ARM64 container, using the supplied SDK:

```bash
cmake -S /pbr/source -B /pbr/build \
  -DPBR_BUNDLE=/pbr/bundle -DCUDAToolkit_ROOT=/usr/local/cuda-13.3 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /pbr/build -j2
cmake --install /pbr/build --prefix /pbr/rebuilt
```

Use `/pbr/rebuilt/bin/benchmark_et` or `benchmark_trt` for individual runs, with
`LD_LIBRARY_PATH=/pbr/bundle/lib` plus the CUDA SDK library directory. Rebuilt
binaries are a separate result cohort: do not silently overwrite the checked
bundle or regenerate its manifest to hide an artifact change.

`geometry_plugin.cu` contains the fixed-shape BEV camera projection and temporal
rotation implementation used by the ONNX engine. It can be rebuilt separately:

```bash
/usr/local/cuda-13.3/bin/nvcc -std=c++17 -O2 --fmad=false \
  -Xcompiler=-fPIC -shared -arch=sm_110 \
  -I/usr/include/aarch64-linux-gnu \
  -I/usr/local/cuda-13.3/thor/targets/aarch64-linux/include \
  /pbr/source/geometry_plugin.cu -lnvinfer -o /pbr/rebuilt/libbev_geometry.so
```

The native SDK is a frozen binary dependency, identified by the artifact manifest.
This command rebuilds the applications, not ExecuTorch or Torch-TensorRT itself.
Use the supplied bundle for the requested benchmark; changing SDK versions,
precision, engine tactics, or geometry changes the comparison.

## Files and provenance

| File | Purpose |
|---|---|
| `benchmark.h` | Shared fixture reader, pinned buffers, timing, correctness, and memory collection |
| `tensorrt.cpp` | Loads ONNX-built engines and runs `enqueueV3`; loads the BEV geometry plugin first |
| `executorch.cpp` | Loads the program, binds caller buffers and stream, and calls `Module::forward` |
| `eager.py`, `models/` | Original FP32 inference baseline and model adapters |
| `run.py` | Verifies artifacts, runs fresh processes, and aggregates valid results |
| `test_run.py` | Tests artifact verification and rejection of incomplete measurements |
| `CMakeLists.txt` | Builds two separate native executables |
| `PBR.md`, `VALIDATION.md`, `THIRD_PARTY.md` | Request, verification record, and source attribution |

Run the host-independent checks with:

```bash
python3 -m unittest discover -s "$PBR_ROOT/source" -p test_run.py -v
```
