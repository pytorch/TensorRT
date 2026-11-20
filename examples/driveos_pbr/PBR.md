# Benchmark request: native perception deployment on DRIVE Thor U

**Request:** Independently reproduce inference latency and memory for
BEVFormer-Tiny and SegFormer-B0 on 14-SM DRIVE Thor U Linux, comparing PyTorch
eager, ONNX–TensorRT C++, and ExecuTorch C++.

**Why:** Perception models can combine learned TensorRT regions with geometry
outside TensorRT. The evaluation compares the deployment cost of a native
TensorRT application, including a custom geometry plugin where needed, with
an ExecuTorch program that coordinates TensorRT and AOTInductor regions.
SegFormer provides a fully TensorRT-backed control. The request is to measure
the tradeoffs, not to establish that one implementation always wins.

**Source:** `pytorch/TensorRT`, branch `abose/PBR-file-driveOS`, directory
`examples/driveos_pbr`. Record the exact source commit in the filed request.

**Attachments:** `driveos-pbr-thoru-20261120.tar.gz` and its SHA-256 file.
This single artifact archive accompanies the request; the Git branch intentionally does not contain engines, weights, or compiled libraries.
Insert the team's artifact links when filing.

**Target:** 14-SM Thor U, DRIVE OS 7.2.5.0 Linux, CUDA 13.3, native TensorRT
11.0.1.7. Use the pinned source builds and supplied checkpoint/fixture bundle. This is
not a QNX request or a Thor X measurement.

| Model | Input workload | Native ONNX path | Native ExecuTorch path |
|---|---|---|---|
| BEVFormer-Tiny | Six 480×800 views plus calibration, motion, previous BEV, and history flag | TensorRT engine with a fixed-shape CUDA geometry plugin | TensorRT engine regions plus AOTInductor geometry |
| SegFormer-B0 | One 1024×1024 RGB tensor | TensorRT engine | TensorRT-backed program |

**Build:** Run `build_native.sh` to compile the native ExecuTorch runtime, CUDA
backend, TensorRT delegate and benchmark applications from the pinned source
dependencies. Run `build_authoring.sh` to build the Torch-TensorRT Python compiler,
then `export_pte.py` for each model. Select the rebuilt runtime with `--native-dir`
and fresh programs with `--program-dir`. The ONNX engines and pinned Python
dependencies are supplied. No separate Bazel step is required for this path.

**Protocol:** Two fixed snapshots per model, three fresh launches per snapshot
and mode; one first call, 20 warmup calls, and 100 measured calls per process.
Each path uses FP32 artifacts and the same input/reference tensors. The timing
boundary includes transfers from CPU inputs through completed CPU outputs.
Check every output at `atol=rtol=3e-4` before and after timing. Native paths use
registered reusable CPU buffers. Do not enable CUDA Graph replay, shared scratch,
or mixed precision for this request. Keep the target otherwise idle and record
power/clock settings.

**Requested deliverables:**

1. Mean and range of six process-median latencies, with all raw call timings.
2. Loading peak RSS, retained RSS/PSS after inference, lifetime peak RSS, and
   CUDA free-memory decrease, reported separately in MiB.
3. Output comparison reports and loaded-library maps for both native applications.
4. Exact board/software versions, source commit, artifact checksums, and any
   difference from the supplied configuration.

**Completion criteria:** All 36 processes complete with all correctness checks
passing and raw evidence attached. Report regressions or failures as observed;
there is no required performance ordering. ONNX and Torch-TensorRT build
different engines, so their difference does not isolate runtime overhead.

**Reproduction:** Follow [README.md](README.md), starting with artifact checksum
verification. [VALIDATION.md](VALIDATION.md) records the handoff test. This request
measures fixed-input inference, not nuScenes detection accuracy or Cityscapes
segmentation accuracy.
