# Handoff validation — November 17, 2026

## Checkpoint-to-program export verification — November 20, 2026

A fresh ARM64 NGC 26.06 container built the Torch-TensorRT Python compiler
from `202fba495341f83c4d25b6b2dece95e0242c5369` using `PYTHON_ONLY=1`.
`build_authoring.sh` completed, and both models exported from the supplied
checkpoints using `export_pte.py`. PyTorch and ExecuTorch Python packages used
the pinned wheels; this was not a PyTorch source build.

- Both eager and compiled models matched both supplied reference snapshots at
  `atol=rtol=3e-4` before saving.
- SegFormer produced a `.pte` with `TensorRTBackend`. BEVFormer produced a
  `.pte` plus `aoti_cuda_blob.ptd`, with `TensorRTBackend` and `CudaBackend`.
- Exported-plan checks confirmed caller-owned CPU input/output buffers. The
  exporter refreshes output tensor metadata before memory planning to account
  for the pinned ExecuTorch device-to-host-copy insertion behavior. It does not
  patch an installed dependency or edit the serialized program afterward.
- All 12 fresh native processes passed: two models × two snapshots × three
  launches, using the rebuilt runtime and `--program-dir` to select fresh exports.
  Each ran one first call, 20 warmups and 100 measured calls; every output
  matched at all three checkpoints. Loaded-library audits passed without
  Python, LibTorch or libc10. No export process ran during this final cohort.
- [export_reference.json](export_reference.json) records package versions,
  output errors, delegate IDs, and hashes of the newly generated programs.

These checks validate the new exports in the native runtime. They are not a new
three-mode comparison: eager and ONNX paths were tested in the separate
36-process source-build cohort below. Do not combine those timings with this
export-validation cohort to form comparative bars.

The fresh-container recipe includes the two DRIVE device mappings, unlimited
memlock and `SYS_NICE`. GPU identification alone succeeded without the complete
setup, but allocating a CUDA tensor failed. The complete documented configuration
passed allocation and export checks.

The November 20 artifact archive adds the ExecuTorch, ruamel.yaml, and
flatbuffers export wheels. Every original bundle file was checked against its
manifest and retained unchanged. Fresh programs remain outside the bundle.

## Full native source-build verification — November 20, 2026

`build_native.sh` completed from fresh dependency checkouts in the ARM64 NGC
26.06 container on the same 14-SM DRIVE Thor U. It compiled the ExecuTorch
runtime, CUDA backend, TensorRT delegate and both benchmark applications;
it did not link the bundle's frozen SDK archives.

- Torch-TensorRT runtime source: `202fba495341f83c4d25b6b2dece95e0242c5369`.
- ExecuTorch source: `a9665e25097b324a7116efa2bd6602287c9c064f`.
- The complete 36-process matrix passed with `--native-dir` selecting the new
  installation. Both models, both snapshots and all three execution paths
  passed the first-call, post-warmup and post-measurement output checks.
- Each process ran 20 warmups and 100 measured calls. Native library-map checks
  passed. All 12 ExecuTorch processes loaded the rebuilt `libaoti_cuda_shims.so`
  and `libextension_cuda.so`, without Python, LibTorch or libc10.
- Six host-independent tests, shell syntax and source whitespace checks passed.

Mean process-median latency in this separate source-build cohort was
267.76 / 138.59 / 139.04 ms for BEVFormer-Tiny and
89.27 / 48.67 / 48.70 ms for SegFormer-B0, in eager / TensorRT / ExecuTorch order.
The original reference table below remains the prepared-binary cohort.

This validates the pinned native runtime sources with the supplied models and
Python environment. It does not validate the latest upstream backend ABI,
rebuild Python packages, or regenerate engines/programs.

## Rebase verification — November 19, 2026

The handoff was rebased without conflicts onto `origin/main` at
`13658aa29828e09c514f2855946faa7b719f5a4b`. The rebased implementation commit
was `0887f819b0`; the benchmark source was unchanged from the original handoff.
The subsequent documentation update clarifies the relationship to the blog.

- All five host-independent tests and Python syntax/whitespace checks passed.
- Every file in the supplied artifact bundle passed checksum verification.
- Both C++ applications were rebuilt in the ARM64 NGC container with the
  supplied native SDK and target CUDA 13.3.
- Twelve fresh-process checks passed on 14-SM DRIVE Thor U: both models,
  both input snapshots, and all three execution paths. Native checks used the
  newly rebuilt applications; eager checks used the pinned Python environment.
- Each process completed one first call, 20 warmups and 100 measured calls,
  with all output comparisons passing at `atol=rtol=3e-4`. Native dependency
  checks passed without Python, LibTorch or libc10 loaded.

These checks use one launch per snapshot/path, rather than the three-launch
measurement protocol below. They verify functionality and do not replace the
reference benchmark cohort. The SDK, engines, programs and geometry plugin
remain the frozen supplied artifacts: this rebase does not validate a rebuild
of the latest upstream TensorRT backend or ExecuTorch runtime.

## Original handoff validation

Both native applications were built from this directory against the relocated
artifact bundle on DRIVE Thor U. The eager baseline used a newly created Python
environment inside the existing NGC 26.06 container. All **36 measured processes**
passed: two models, three modes, two snapshots, and three launches per snapshot.
Each process completed the first call, 20 warmup calls, and 100 measured calls.
Every output passed at `atol=rtol=3e-4` after the first call, after warmup, and
after measurement. Native library-map checks found no Python, LibTorch, or libc10.

The native device query reported Thor, 14 SMs, compute capability 11.0, and CUDA
driver/runtime API version 13030. The target root-filesystem version file reported
`7.2.5.0-46229577`; the OS was Ubuntu 24.04.4 LTS, ARM64. No clock, power-mode,
global-cache, or installed system-runtime changes were made for this matrix.

## Results from this handoff

Values below are means across six processes. Each latency value is the mean of
six process medians, each over 100 calls. [reference_summary.csv](reference_summary.csv)
also includes the process minima/maxima and lifetime peak RSS.

| Model | Mode | Latency (ms) | Loading peak RSS (MiB) | Retained RSS (MiB) | Retained PSS (MiB) | CUDA free-memory decrease (MiB) |
|---|---|---:|---:|---:|---:|---:|
| BEVFormer-Tiny | PyTorch eager | 267.70 | 1048.23 | 1760.95 | 1707.17 | 1542 |
| BEVFormer-Tiny | ONNX–TensorRT C++ | 138.53 | 548.10 | 453.01 | 409.13 | 590 |
| BEVFormer-Tiny | ExecuTorch C++ | 139.02 | 586.52 | 483.93 | 437.91 | 622 |
| SegFormer-B0 | PyTorch eager | 89.29 | 856.03 | 1573.80 | 1520.00 | 772 |
| SegFormer-B0 | ONNX–TensorRT C++ | 48.63 | 411.34 | 434.23 | 390.41 | 628 |
| SegFormer-B0 | ExecuTorch C++ | 48.72 | 428.57 | 433.66 | 390.25 | 612 |

These are reference observations, not acceptance thresholds or guaranteed
performance. ONNX and ExecuTorch use independently built engines. Memory metrics
are separate views and must not be summed.

The clean TensorRT runner releases serialized engine bytes immediately after
deserialization, before creating the execution context. Its BEVFormer loading
peak is therefore lower than in the earlier diagnostic runner, which retained
that buffer longer. Use this table for this handoff; do not mix it with older
loading-peak figures. Eager process memory also reflects the fresh environment.

## Reproduction checks

- CMake configured, compiled, and linked both native applications on ARM64 using
  the supplied SDK and CUDA 13.3.
- Both models passed all native and eager comparisons on both snapshots.
- Five host-independent tests passed, covering artifact checksums, path boundaries,
  missing timing samples, missing/failed correctness checks, and summary arithmetic.
- The bundled BEV plugin source compiled with the documented `nvcc` command;
  the rebuilt plugin passed both fixtures in additional correctness runs.
- Python files passed syntax checks; source whitespace checks passed.

The initial fresh-environment test identified two dependencies absent from a
plain Torch-wheel install: CUDA Python libraries and matching TorchVision.
The handoff includes both, and the README includes their installation steps.
TorchVision `0.30.0.dev20261007+cu132` was repacked as a wheel from the previously
validated installation, preserving its source/binary files and license.
SegFormer was rerun after that correction; all 18 SegFormer processes passed.
The 18 BEVFormer runs had already passed and do not import TorchVision.

The results attachment combines those two successful cohorts. Original per-cohort
configuration records are preserved. The final artifact manifest adds the
TorchVision wheel and retains the earlier manifest under
`provenance/bevformer-run-manifest.json`; model, fixture, native library, and
benchmark binary hashes are unchanged between the cohorts.

## Artifacts to attach

- `driveos-pbr-thoru-20261117.tar.gz` and its `.sha256` file: prepared artifacts,
  native SDK, authoring dependencies, and license notices.
- `driveos-pbr-validation-20261117.tar.gz`: the 36 successful processes' raw timing,
  correctness, memory, library-map, command, and configuration records.
- Exact source commit from `abose/PBR-file-driveOS`.

Final `bundle/manifest.json` SHA-256:

```text
58a9068731d06cfa931f43740d04a7fbe2478ccba90be6e5bf00e2bba5fae727
```

The original handoff rebuilt benchmark applications against the frozen native
SDK. The source-build procedure added on November 20 also rebuilds the pinned
native runtime and delegate. Neither procedure rebuilds the Python compiler
packages, re-exports the models on another TensorRT release, or validates a newly
flashed target setup. The PBR has been prepared for filing; submission and
attachment hosting are separate steps.
