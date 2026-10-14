"""Numerical and export gates for a pretrained BEVFormer attention component.

The independent reference executes upstream forward methods on CPU with
MMCV's PyTorch attention reference. It does not validate MMCV's CUDA binary.
Only the selected function ASTs are executed, avoiding old training imports.
"""

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import MethodType

import torch
from attention import example_inputs, load_attention
from torch.nn import functional as F


def extract_function(path, name, namespace, class_name=None):
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name:
        body = next(
            node.body
            for node in body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    function = next(
        node for node in body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    # The upstream force_fp32 decorator only handles dtype conversion here.
    # Test inputs and weights are explicitly FP32, in evaluation mode.
    function.decorator_list = []
    # Execute only reference functions from the hash-verified upstream source.
    exec(  # noqa: S102
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[name]


def make_reference(model, root):
    namespace = {"torch": torch, "F": F}
    mmcv = root / "upstream/mmcv/multi_scale_deform_attn.py"
    extract_function(mmcv, "multi_scale_deformable_attn_pytorch", namespace)
    source = (
        root
        / "upstream/BEVFormer/projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py"
    )
    reference = copy.deepcopy(model).cpu()
    deform_forward = extract_function(
        source, "forward", namespace, "MSDeformableAttention3D"
    )
    reference.deformable_attention.forward = MethodType(
        deform_forward, reference.deformable_attention
    )
    spatial_forward = extract_function(
        source, "forward", namespace, "SpatialCrossAttention"
    )
    reference.forward = MethodType(spatial_forward, reference)
    return reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--queries", type=int, default=2500)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    sources = json.loads((args.root / "sources.json").read_text())
    checkpoint = args.root / "checkpoints/bevformer_tiny_epoch_24.pth"
    spatial_source = (
        args.root
        / "upstream/BEVFormer/projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py"
    )
    for path, expected_hash in [
        (checkpoint, sources["checkpoint_sha256"]),
        (spatial_source, sources["spatial_attention_source_sha256"]),
        (
            args.root / "upstream/mmcv/multi_scale_deform_attn.py",
            sources["mmcv_reference_sha256"],
        ),
    ]:
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected_hash:
            raise ValueError(f"Source integrity mismatch: {path}")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_attention(checkpoint, layer=args.layer)
    reference = make_reference(model, args.root)
    report = {
        "scope": "pretrained spatial-cross-attention component only",
        "layer": args.layer,
        "queries": args.queries,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "cases": [],
    }
    with torch.inference_mode():
        for seed, mask_mode in [
            (7, "mixed"),
            (19, "camera_missing"),
            (23, "all_visible"),
        ]:
            inputs = example_inputs(args.queries, seed=seed)
            if mask_mode == "camera_missing":
                inputs[-1][0] = 0
                inputs[-1][:, :, :3] = 0  # Queries visible in no camera.
            if mask_mode == "all_visible":
                inputs[-1].fill_(1)
            expected = reference(
                query=inputs[0],
                key=inputs[1],
                value=inputs[1],
                reference_points_cam=inputs[2],
                bev_mask=inputs[3].bool(),
                spatial_shapes=torch.tensor([[15, 25]]),
                level_start_index=torch.tensor([0]),
            )
            actual = model(*inputs)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
            gpu_model = copy.deepcopy(model).cuda()
            cuda_inputs = tuple(x.cuda() for x in inputs)
            cuda_output = gpu_model(*cuda_inputs).cpu()
            torch.testing.assert_close(cuda_output, expected, rtol=2e-4, atol=2e-4)
            report["cases"].append(
                {
                    "seed": seed,
                    "visibility": mask_mode,
                    "cpu_max_abs_error": (actual - expected).abs().max().item(),
                    "cuda_max_abs_error": (cuda_output - expected).abs().max().item(),
                }
            )
            print("PASS:", report["cases"][-1], flush=True)
        # All-invisible is also defined in our fixed-shape interface.
        zero_mask_inputs = list(example_inputs(args.queries))
        zero_mask_inputs[-1].zero_()
        expected = zero_mask_inputs[0] + model.output_proj.bias
        torch.testing.assert_close(model(*zero_mask_inputs), expected)
        exported = torch.export.export(gpu_model, cuda_inputs, strict=True)
        # Different values and visibility after export detect accidental specialization.
        alternate = tuple(x.cuda() for x in example_inputs(args.queries, seed=51))
        torch.testing.assert_close(exported.module()(*alternate), gpu_model(*alternate))
        report["strict_export"] = "PASS"
        report["exported_call_functions"] = sorted(
            {str(n.target) for n in exported.graph.nodes if n.op == "call_function"}
        )
        report["all_invisible"] = "PASS"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: reference equivalence and strict export;", args.report)


if __name__ == "__main__":
    main()
