# Handoff validation — November 17, 2026

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

The handoff rebuilds benchmark applications against the frozen native SDK. It
does not establish a from-scratch compiler/runtime build, re-export on another
TensorRT release, or a newly flashed/containerized target setup. The PBR has been
prepared for filing; submission and attachment hosting are separate steps.
