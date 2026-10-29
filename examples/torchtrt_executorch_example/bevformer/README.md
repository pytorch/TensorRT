# BEVFormer Tiny on DRIVE OS

This experimental example adapts the official pretrained BEVFormer-Tiny model
for Torch-TensorRT export and native ExecuTorch deployment. It replaces MMCV's
custom deformable-attention extension with PyTorch sampling operations and
makes camera calibration, motion metadata and previous-frame BEV state explicit
tensor inputs. It does not require installing the old training framework.

The model runs in FP32 evaluation mode, batch one, with six preprocessed
480-by-800 camera images. Synthetic-input correctness tests are not a nuScenes
detection-accuracy result or a performance benchmark.

## Current validation

Validation uses DRIVE OS 7.2.5 Thor, CUDA 13.3 and native TensorRT 11.0.1.7,
with the existing NGC 26.06 authoring environment. PyTorch is the 2.16 cu132
nightly; TensorRT Python bindings report 11.0.0.114. These results do not
establish compatibility with a different DRIVE release.

| Check | Result |
|---|---|
| Pretrained spatial-attention component, TRT-only `.pte` | Python numerical checks and native host smoke passed |
| Same component, deliberately mixed TRT/AOTI `.pte` | Python numerical checks and native host smoke passed |
| Full Tiny checkpoint | All 643 state tensors loaded strictly |
| Full tensor core versus upstream forward methods | Passed with first-frame and history-enabled synthetic inputs |
| Strict PyTorch export of the full tensor core | Passed; changed values and history flag also passed |
| TensorRT compilation of isolated learned regions | Backbone, temporal/spatial attention and decoder attention checks passed |
| Full graph with geometry outside TensorRT | Numerical comparison passed |
| Full-model `.pte` | Mixed TensorRT/CUDA export passed with reshape-copy normalization and eager division rounding |
| Full-model native deployment | All-output comparisons and persistent recurrence passed outside Docker |

For the full-forward comparison, the largest absolute differences observed
were below 7e-6 for class logits, 4e-5 for box predictions and 1.1e-5 for the
next BEV state. The reference executes pinned upstream backbone, encoder,
decoder and head methods, with MMCV's independent PyTorch sampler in place of
its compiled extension. It is not a comparison against the MMCV CUDA binary.

The later full-model study passed 12 synthetic frames over three scenes with
native recurrence and explicit resets (7,896,000 output values, zero mismatches
at `atol=rtol=3e-4`). The standalone C++ decoder matched the pinned upstream
oracle on those outputs and boundary/empty cases. These are historical results
on the October 2026 authoring stack, not a new hardware run of this rebased
branch. No real driving-sequence or nuScenes mAP/NDS result is established.

## Files and data

| File | Purpose |
|---|---|
| `model.py` | Full Tiny tensor core, checkpoint loading and synthetic inputs |
| `attention.py` | Fixed-shape spatial cross-attention implementation |
| `validate_model.py` | Full upstream-reference and strict-export checks |
| `export_full.py` | Reproducible mixed export, `.ptd` loading and all-output checks |
| `prepare_native_cases.py` | Independent or changing-scene synthetic reference fixtures |
| `decode_detections.cpp`, `check_detection_decoder.py` | Native box decoding and pinned upstream comparison |
| `export_model.py`, `diagnose_compilation.py` | Optional region-selection diagnostics |
| `../native_validation/` | Native correctness/recurrence runner and memory/latency tools |
| `validate_attention.py`, `export_attention.py` | Smaller attention-only tests |
| `prepare_assets.py` | Download or verify pinned sources and checkpoint |
| `sources.json`, `reference_sources.json` | Revisions and SHA-256 hashes |

Source lives in this example directory. Keep downloaded references, checkpoint,
generated models and logs in a separate data directory. The `--root` argument
selects that directory; the Python implementation embeds no workspace or SDK
installation paths.

From this example directory, in the configured DRIVE authoring environment:

```bash
bevformer_data=/path/to/bevformer-data
python prepare_assets.py --root "$bevformer_data" --checkpoint
```

Replace `/path/to/bevformer-data` with your chosen location. The checkpoint is
approximately 383 MB. Existing files are checked rather than silently replaced;
the checkpoint is loaded with `weights_only=True` and strict state matching.

## Full model interface

```text
Preprocessed six-camera images + calibration + CAN-bus metadata
                             + previous BEV + history-valid flag
                                      │
                               ResNet-50 and FPN
                                      │
                         Three temporal/spatial encoder layers
                                      │
                              Six decoder layers
                                      │
                   Class logits + raw boxes + next BEV state
```

| Input | Shape | Contract |
|---|---|---|
| `images` | `[1,6,3,480,800]` | FP32, already normalized, resized and padded |
| `lidar2img` | `[1,6,4,4]` | FP32 projection matrices adjusted for the image resize |
| `can_bus` | `[1,18]` | Upstream metadata convention: translation delta in the first entries, absolute yaw in radians at `-2`, yaw delta in degrees at `-1` |
| `prev_bev` | `[1,2500,256]` | Previous output BEV; zeros are suitable when history is invalid |
| `has_prev` | `[1,1,1]` | FP32 zero for a scene's first frame, one when history is valid |

Outputs are class logits `[1,900,10]`, raw box predictions `[1,900,10]`, and
next BEV `[1,2500,256]`. Box entries are `(cx,cy,log_w,log_l,cz,log_h,sin_yaw,
cos_yaw,vx,vy)`, with centers mapped to the configured world-coordinate range.
The caller must perform score selection and box decoding before treating these
as detections.

The caller also owns scene reset and CAN-bus pose-difference bookkeeping. This
port preserves upstream's rotation center `[100,100]`, including on Tiny's
50-by-50 BEV grid. Dense spatial attention evaluates all camera/query pairs
before masking; this can cost more computation and memory than upstream sparse
rebatching.

## Validate before compiling

```bash
python validate_model.py --root "$bevformer_data" \
  --report "$bevformer_data/logs/full-model-validation.json"
```

Use `--diagnose` to print intermediate-layer errors, or `--skip-export` to run
only the upstream-reference comparison. Diagnostic hooks are removed before
export. Both history branches are tested after export with changed inputs.

## Export and check the complete program

The maintained full-model path assigns learned regions to TensorRT and the
complete `Geometry` module to CUDA/AOTInductor. The branch normalizes
`aten::_reshape_copy` to reshape plus a contiguous clone before ExecuTorch
partitioning. `export_full.py` explicitly enables
`eager_numerics.division_rounding=True` while compiling AOTI geometry, because
small projection differences can change camera-visibility masks. It checks all
three outputs for reset/history inputs and retains the same 3e-4 tolerances.

The original all-TensorRT geometry discrepancy remains unresolved; this routing
is a numerical workaround for the tested model, not evidence that TensorRT
cannot implement geometry or grid sampling in general.

```bash
python export_full.py \
  --checkpoint "$bevformer_data/checkpoints/bevformer_tiny_epoch_24.pth" \
  --output-dir "$bevformer_data/artifacts/full-mixed"
python prepare_native_cases.py \
  --checkpoint "$bevformer_data/checkpoints/bevformer_tiny_epoch_24.pth" \
  --output-dir "$bevformer_data/artifacts/cases" \
  --scenes 3 --frames-per-scene 4 --seed 101
```

Output directories must be new. Export writes `model.pte`,
`aoti_cuda_blob.ptd` and `report.json`. Keep both model files together and require
`FULL BEVFORMER EXPORT PASS`; a partially written artifact is not a passing run.
The report records versions, hashes, backend IDs and every output comparison.
The fixture generator writes `sequence.txt` plus raw FP32 input/reference files
and shape sidecars. Synthetic scenes change motion and history but are not
recorded camera data.

Build and stage the applications following
[`native_validation/README.md`](../native_validation/README.md). Run its
`validate_pte --bev-sequence` command on the DRIVE host. It carries its own
previous-BEV output between frames, loads the model once and resets state at
scene boundaries. It does not require Python or LibTorch. Audit `ldd` and the
saved `loaded-maps.txt` files, including dynamically loaded AOTI libraries.

To decode a saved frame's raw boxes:

```bash
c++ -std=c++17 -O2 -Wall -Wextra -Werror decode_detections.cpp -o "$decoder"
"$decoder" "$native_results/frame_0" "$decoded_tsv"
python check_detection_decoder.py \
  --upstream "$bevformer_data/upstream/BEVFormer" --decoder "$decoder" \
  --raw-dir "$native_results" --output-dir "$decoder_checks" --include-synthetic
python test_decoder_failures.py "$decoder"
```

The decoder is CPU-only C++ and writes scored class/3D-box records. The Python
oracle is an offline test, not a deployment dependency. Synthetic predictions
are not evidence of observed vehicles or detector accuracy. `prepare_assets.py`
also fetches the hash-pinned decoder-reference sources used by these checks.

`export_model.py --cuda-geometry` remains available for the earlier diagnostic
workflow; `--cuda-grid` forces sampling out of TensorRT for experimentation.
Use `export_full.py` for the documented reproducible bundle and
`prepare_native_cases.py` for the native fixtures.

The shared benchmark compares eager, Python ExecuTorch/Torch-tensor and native
C++ execution of matching inputs. Its RSS/PSS are process-memory measurements,
not total GPU memory. See the shared README for timing boundaries and fresh
process repetitions. No hardware performance numbers are regenerated by this
branch integration.

## Source provenance

- [BEVFormer](https://github.com/fundamentalvision/BEVFormer), revision
  `66b65f3a1f58caf0507cb2a971b9c0e7f842376c`, official Tiny checkpoint.
- MMDetection v2.14.0 and MMCV v1.4.0 forward-method references.
- TorchVision v0.10.1 rotation-coordinate reference; BSD notice in
  `LICENSE.torchvision`.

BEVFormer/OpenMMLab attribution and Apache-2.0 licensing are retained in the
source and `LICENSE`. Check the upstream dataset and checkpoint terms before
using them beyond this validation example.
