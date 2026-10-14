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
| Full-model `.pte` | Blocked by missing AOTI fallback for `aten::_reshape_copy` |
| Full-model native deployment | Not yet validated |

For the full-forward comparison, the largest absolute differences observed
were below 7e-6 for class logits, 4e-5 for box predictions and 1.1e-5 for the
next BEV state. The reference executes pinned upstream backbone, encoder,
decoder and head methods, with MMCV's independent PyTorch sampler in place of
its compiled extension. It is not a comparison against the MMCV CUDA binary.

## Files and data

| File | Purpose |
|---|---|
| `model.py` | Full Tiny tensor core, checkpoint loading and synthetic inputs |
| `attention.py` | Fixed-shape spatial cross-attention implementation |
| `validate_model.py` | Full upstream-reference and strict-export checks |
| `export_model.py` | Full-model TensorRT compilation, `.pte` export and numerical checks |
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

Current blocker: normal TensorRT selection changes camera projection and BEV
rotation results in the tested inputs. `--cuda-geometry` keeps those operations
outside TensorRT, allowing the compiled graph's numerical comparison to pass.
The subsequent ExecuTorch/AOTI lowering currently fails with
`Method forward missing fallback kernels: aten::_reshape_copy`. No passing
full-model `.pte` has been produced; this must be resolved before native testing.

Use a fresh output directory for every attempt:

```bash
python export_model.py --root "$bevformer_data" --cuda-geometry \
  --output-dir "$bevformer_data/artifacts/full-cuda-geometry"
```

TensorRT receives supported regions; ExecuTorch's CUDA backend compiles remaining
supported regions using AOTInductor. `--cuda-geometry` excludes the complete
`Geometry` module by its exported module type. `--cuda-grid` deliberately excludes
`grid_sample` from TensorRT to exercise a mixed path; it is not evidence that
TensorRT cannot handle that operation.

To isolate conversion failures without generating a deployment artifact:

```bash
python diagnose_compilation.py --root "$bevformer_data" \
  --report "$bevformer_data/logs/trt-regions.json"
```

This records per-region correctness results. Check each region's `pass` field;
the diagnostic script completes even when a region fails.

Require the final `PASS` and `validation.json`; a directory or partially written
`.pte` alone does not mean the test passed. Preserve all generated `.ptd` files
beside their `.pte`. The script compares all three outputs using ExecuTorch's
Python runtime and writes raw FP32 input/reference files for native validation.

The generic native reference runner currently fills inputs with ones. That was
adequate for the component smoke test, but it is not a real-camera full-model
test. File-based native input/output validation, temporal sequence testing,
image preprocessing and final detection decoding remain follow-up work.

## Source provenance

- [BEVFormer](https://github.com/fundamentalvision/BEVFormer), revision
  `66b65f3a1f58caf0507cb2a971b9c0e7f842376c`, official Tiny checkpoint.
- MMDetection v2.14.0 and MMCV v1.4.0 forward-method references.
- TorchVision v0.10.1 rotation-coordinate reference; BSD notice in
  `LICENSE.torchvision`.

BEVFormer/OpenMMLab attribution and Apache-2.0 licensing are retained in the
source and `LICENSE`. Check the upstream dataset and checkpoint terms before
using them beyond this validation example.
