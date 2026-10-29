# SegFormer-B0 on DRIVE OS

This example exports a pretrained street-scene segmentation model through
Torch-TensorRT into an ExecuTorch `.pte`, then validates it in a native C++
application on DRIVE Thor. All model regions use the TensorRT delegate; this
example does not require AOTInductor or an external `.ptd` weights file.

The Python wrapper uses the maintained Hugging Face Transformers implementation.
It returns raw logits without reimplementing the architecture. Image decoding,
resizing, normalization and segmentation visualization remain outside the
exported model.

```text
RGB image → resize and normalize in Python
                         │
                         ▼
              SegFormer-B0 pretrained model
              hierarchical encoder + decode head
                         │ Torch-TensorRT compilation
                         ▼
              model.pte / TensorRTBackend
                         │
                         ▼
              Native ExecuTorch C++ application
                         │
                         ▼
              Raw segmentation logits
              compare against offline PyTorch reference
```

## Provenance and validation scope

- Implementation: Transformers `4.55.4`, `SegformerForSemanticSegmentation`.
- Checkpoint: [NVIDIA SegFormer-B0 Cityscapes](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024),
  revision `21b3847fae21ddee674abd31129307b6a1235bd9`.
- Weights are checked by SHA-256 and loaded with `strict=True` and
  `weights_only=True`. No checkpoints, images or exported binaries are committed.
- Checkpoint terms are documented in its model card and the
  [SegFormer license](https://github.com/NVlabs/SegFormer/blob/master/LICENSE).
- Default input: FP32 `[1,3,1024,1024]`, RGB with ImageNet normalization.
  Output: FP32 logits `[1,19,256,256]`; TF32 is disabled.
- The historical preprocessor configuration defaults to 512. This example
  explicitly overrides image size to match the exported tensor interface.

The October 2026 study passed full-output comparisons for three synthetic
fixtures and two road images on DRIVE OS 7.2.5 Thor, with CUDA 13.3 and native
TensorRT 11.0.1.7. Authoring used NGC 26.06, PyTorch/ExecuTorch cu132 nightlies,
and TensorRT Python package 11.0.0.114. Native process maps contained no
Python/LibTorch. These are historical results, not a new hardware validation of
this rebased branch. Agreement with PyTorch is not a labelled Cityscapes mIoU
result; dataset accuracy remains unmeasured.

## Export in the authoring environment

Use the DRIVE support branch's configured authoring environment, matching the
target's SDK and the chosen ExecuTorch exporter/runtime revision. It needs
Torch-TensorRT's Python export support, ExecuTorch and the TensorRT delegate
runtime for the Python correctness check. Python-only Torch-TensorRT export
does not mean Python without PyTorch.

`requirements-authoring.txt` records the additional model-package versions used
in the study. Install them into the configured authoring environment while
preserving its Torch/CUDA/TensorRT dependencies. Inspect pip's proposed changes
before applying them:

```bash
repo=$(git rev-parse --show-toplevel)
example="$repo/examples/torchtrt_executorch_example/segformer"
validation="$repo/examples/torchtrt_executorch_example/native_validation"
python -m pip install --dry-run -r "$example/requirements-authoring.txt"
# After verifying the dependency plan:
python -m pip install -r "$example/requirements-authoring.txt"

python "$example/fetch_checkpoint.py" --output "$checkpoint"
python "$example/export_segformer.py" \
  --checkpoint "$checkpoint" --output "$bundle"
```

Set `$checkpoint` and `$bundle` to new directories of your choice. The exporter
writes three `seed_*` input/reference fixture directories, `model.pte`, and a
report containing versions, hashes, backend IDs and comparisons. It checks
the compiled TensorRT model and the Python ExecuTorch execution, then writes
`EXPORT_PASS`. A partial `.pte` file or successful export alone is not a native
deployment pass. `--eager-only` generates references without exporting a program.

## Validate outside the container

Build and stage `validate_pte` using
[`../native_validation/README.md`](../native_validation/README.md). The native
application uses the ExecuTorch runtime and TensorRT delegate; Python and
LibTorch are not inference dependencies. On the DRIVE host, provide host paths:

```bash
bash "$validation/run_cases_host.sh" \
  "$deployment/bin/validate_pte" "$bundle" "$bundle" \
  "$native_results" "$deployment/lib" "$cuda_root"
```

The same bundle directory holds the `.pte` and its `seed_*` fixtures. The runner
compares every logit at `atol=rtol=3e-4`, records raw native outputs, and checks
linked and dynamically loaded libraries. Inspect its final PASS and output
files before proceeding to a different input.

## Real-image validation

Supply an image and a provenance JSON with its `sha256`, `source_url`,
`license` and `author`. The script checks the image hash; it does not
download or redistribute images. Choose a new `$fixtures/road` directory:

```bash
python "$example/validate_real_image.py" prepare \
  --image "$image" --provenance "$provenance" \
  --checkpoint "$checkpoint" --bundle "$bundle" \
  --output "$fixtures/road"
bash "$validation/run_cases_host.sh" \
  "$deployment/bin/validate_pte" "$bundle" "$fixtures" \
  "$real_native_results" "$deployment/lib" "$cuda_root"
python "$example/validate_real_image.py" analyze \
  --fixture "$fixtures/road" --native "$real_native_results/road" \
  --output "$image_comparison"
```

Prepare/analyze run offline in Python; native inference runs on the host.
Preparation resizes the entire frame without a crop, verifies normalization
against the recorded processor settings, and generates the eager reference.
Analysis compares logits and class maps and writes labelled visualizations.
It interpolates logits before argmax, never class IDs. No human annotations are
needed for this numerical check, so it cannot establish task accuracy.

## Labelled accuracy protocol

`evaluate_cityscapes.py` supports a caller-provided validation dataset with
paired RGB images and masks. It records the subset, preprocessing, label
mapping, checkpoint and artifact hashes. This optional protocol has software
tests but no measured dataset accuracy in this study.

```bash
python "$example/evaluate_cityscapes.py" prepare \
  --data-root "$cityscapes" --checkpoint "$checkpoint" --bundle "$bundle" \
  --output "$accuracy_fixtures" --mask-encoding label_ids --limit 10 --seed 0
bash "$validation/run_cases_host.sh" \
  "$deployment/bin/validate_pte" "$bundle" "$accuracy_fixtures" \
  "$accuracy_native" "$deployment/lib" "$cuda_root"
python "$example/evaluate_cityscapes.py" score \
  --fixtures "$accuracy_fixtures" --native "$accuracy_native" \
  --output "$accuracy_report"
```

Use `--limit 0` for all available validation images. Report subset size and
resize protocol alongside mIoU; this is not an official benchmark submission.

## Latency and memory

Use the shared benchmark tools with `--model-family segformer`, the same
checkpoint and fixture, and an existing `.pte`. Compare eager PyTorch,
Python/Torch ExecuTorch, and native C++ in separate fresh processes. Time units
are milliseconds; RSS, PSS and peak RSS are process-memory measures, not total
GPU usage. See the shared README for the timing boundaries and limitations.

`summarize_matrix.py` validates the fixed 1024-input study protocol: three
fresh processes per mode, 20 warmups, 100 measured calls, and full-logit checks
at three stages. It expects `eager/repeat-0..2`, `executorch/repeat-0..2`,
`native/repeat-0..2`, and `sha256.txt` recording the `.pte` and input/reference
`.bin`/`.shape` files. Use the general shared summarizers for other protocols;
do not mix old measurements with newly generated artifacts.

## Helper tests

```bash
python -m unittest discover -s "$example" -p 'test_*.py' -v
```

Tests cover logits/shape handling, label mapping, confusion-matrix metrics,
evidence rejection, and benchmark parsing. Dataset pipeline tests use tiny
invented images and a fake model; they are not deployed-model accuracy tests.
