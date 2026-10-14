# SPDX-License-Identifier: Apache-2.0
"""Compare the tensor port with pinned upstream forward methods, then export.

The reference preserves upstream sparse rebatching, metadata flow, temporal
branches and decoder refinement. MMCV's independent PyTorch sampler replaces
its compiled extension. This is NOT a nuScenes accuracy or CUDA-extension test.
No old training-framework imports or unrestricted checkpoint pickle are used.
"""

import argparse
import copy
import hashlib
import json
import math
import warnings
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import List

import numpy as np
import torch
from model import Bottleneck, ResNetC5, example_inputs, load_model
from torch import nn
from torch.nn import functional as F
from validate_attention import extract_function


def make_reference(model, root):
    manifest = json.loads(
        (Path(__file__).parent / "reference_sources.json").read_text()
    )
    for relative, expected in manifest["sha256"].items():
        with (root / relative).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Reference source hash mismatch: {relative}")
    namespace = {
        "torch": torch,
        "F": F,
        "np": np,
        "math": math,
        "copy": copy,
        "warnings": warnings,
        "Tensor": torch.Tensor,
        "List": List,
        "TORCH_VERSION": torch.__version__,
        "digit_version": lambda version: tuple(int(p) for p in version.split(".")[:2]),
    }
    modules = root / "upstream/BEVFormer/projects/mmdet3d_plugin/bevformer/modules"
    mmcv = root / "upstream/mmcv/transformer.py"
    sampler = extract_function(
        root / "upstream/mmcv/multi_scale_deform_attn.py",
        "multi_scale_deformable_attn_pytorch",
        namespace,
    )

    def sample_apply(value, shapes, starts, locations, weights, step):
        return sampler(value, shapes, locations, weights)

    namespace["MultiScaleDeformableAttnFunction_fp32"] = SimpleNamespace(
        apply=sample_apply
    )
    affine = extract_function(
        root / "upstream/torchvision/functional.py",
        "_get_inverse_affine_matrix",
        namespace,
    )
    make_grid = extract_function(
        root / "upstream/torchvision/functional_tensor.py",
        "_gen_affine_grid",
        namespace,
    )

    def rotate(image, angle, center):
        h, w = image.shape[-2:]
        adjusted_center = [center[0] - w * 0.5, center[1] - h * 0.5]
        matrix = affine(adjusted_center, -float(angle), [0.0, 0.0], 1.0, [0.0, 0.0])
        theta = image.new_tensor(matrix).reshape(1, 2, 3)
        grid = make_grid(theta, w, h, w, h)
        return F.grid_sample(image[None], grid, mode="nearest", align_corners=False)[0]

    namespace["rotate"] = rotate
    extract_function(modules / "decoder.py", "inverse_sigmoid", namespace)

    def bind(module, path, class_name, method="forward"):
        function = extract_function(path, method, namespace, class_name)
        setattr(module, method, MethodType(function, module))

    reference = copy.deepcopy(model)
    backbone = reference.img_backbone
    # Properties alias existing BatchNorm modules; no new learned weights.
    for cls in (Bottleneck, ResNetC5):
        cls.norm1 = property(lambda self: self.bn1)
    Bottleneck.norm2 = property(lambda self: self.bn2)
    Bottleneck.norm3 = property(lambda self: self.bn3)
    for block in backbone.modules():
        if isinstance(block, Bottleneck):
            block.with_cp = block.with_plugins = False
            bind(block, root / "upstream/mmdet/resnet.py", "Bottleneck")
    backbone.deep_stem = False
    backbone.res_layers = ["layer1", "layer2", "layer3", "layer4"]
    backbone.out_indices = (3,)
    bind(backbone, root / "upstream/mmdet/resnet.py", "ResNet")
    neck = reference.img_neck
    neck.in_channels, neck.start_level, neck.num_outs = [2048], 0, 1
    bind(neck, root / "upstream/mmdet/fpn.py", "FPN")
    head = reference.pts_bbox_head
    bind(
        head.positional_encoding,
        root / "upstream/mmdet/positional_encoding.py",
        "LearnedPositionalEncoding",
    )
    head.bev_h = head.bev_w = 50
    head.real_h = head.real_w = 102.4
    head.pc_range = [-51.2, -51.2, -5, 51.2, 51.2, 3]
    head.with_box_refine, head.as_two_stage = True, False
    bind(head, modules.parent / "dense_heads/bevformer_head.py", "BEVFormerHead")
    transformer = head.transformer
    transformer.embed_dims = 256
    transformer.rotate_prev_bev = transformer.use_shift = transformer.use_can_bus = True
    transformer.use_cams_embeds = True
    transformer.rotate_center = [100, 100]
    bind(
        transformer,
        modules / "transformer.py",
        "PerceptionTransformer",
        "get_bev_features",
    )
    bind(transformer, modules / "transformer.py", "PerceptionTransformer")
    encoder = transformer.encoder
    encoder.pc_range, encoder.num_points_in_pillar = head.pc_range, 4
    encoder.return_intermediate = False
    encoder.get_reference_points = extract_function(
        modules / "encoder.py", "get_reference_points", namespace, "BEVFormerEncoder"
    )
    bind(encoder, modules / "encoder.py", "BEVFormerEncoder", "point_sampling")
    bind(encoder, modules / "encoder.py", "BEVFormerEncoder")
    decoder = transformer.decoder
    decoder.return_intermediate = True
    bind(decoder, modules / "decoder.py", "DetectionTransformerDecoder")
    for layer in list(encoder.layers) + list(decoder.layers):
        layer.operation_order = (
            "self_attn",
            "norm",
            "cross_attn",
            "norm",
            "ffn",
            "norm",
        )
        layer.num_attn, layer.pre_norm = 2, False
        ffn = layer.ffns[0]
        ffn.dropout_layer, ffn.add_identity = nn.Identity(), True
        bind(ffn, mmcv, "FFN")
    for layer in encoder.layers:
        bind(layer, modules / "encoder.py", "BEVFormerLayer")
        bind(
            layer.attentions[0],
            modules / "temporal_self_attention.py",
            "TemporalSelfAttention",
        )
        spatial = layer.attentions[1]
        bind(spatial, modules / "spatial_cross_attention.py", "SpatialCrossAttention")
        spatial.deformable_attention.im2col_step = 64
        bind(
            spatial.deformable_attention,
            modules / "spatial_cross_attention.py",
            "MSDeformableAttention3D",
        )
    for layer in decoder.layers:
        bind(layer, mmcv, "BaseTransformerLayer")
        self_attention = layer.attentions[0]
        self_attention.batch_first = False
        self_attention.proj_drop = self_attention.dropout_layer = nn.Identity()
        bind(self_attention, mmcv, "MultiheadAttention")
        layer.attentions[1].batch_first = False
        bind(layer.attentions[1], modules / "decoder.py", "CustomMSDeformableAttention")

    def forward(images, lidar2img, can_bus, previous, has_previous):
        features = reference.img_neck(reference.img_backbone(images.flatten(0, 1)))[0]
        metadata = [
            {
                "can_bus": can_bus[0].cpu().numpy(),
                "lidar2img": lidar2img[0].cpu().numpy(),
                "img_shape": [(480, 800, 3)] * 6,
            }
        ]
        history = (
            previous.transpose(0, 1).clone() if bool(has_previous.item()) else None
        )
        outputs = head([features[None]], metadata, prev_bev=history)
        return (
            outputs["all_cls_scores"][-1],
            outputs["all_bbox_preds"][-1],
            outputs["bev_embed"].transpose(0, 1),
        )

    forward.reference_model = reference
    return forward


def diagnostic_hooks(model, captured):
    names = ["img_backbone", "img_neck"]
    for kind, count in (("encoder", 3), ("decoder", 6)):
        for index in range(count):
            prefix = f"pts_bbox_head.transformer.{kind}.layers.{index}"
            names.extend(f"{prefix}.norms.{norm}" for norm in range(3))

    def save(name):
        def hook(_module, _inputs, outputs):
            output = outputs[0] if isinstance(outputs, tuple) else outputs
            if output.ndim == 3 and output.shape[1] == 1:
                output = output.transpose(0, 1)
            captured[name] = output.detach().cpu().clone()

        return hook

    return [
        model.get_submodule(name).register_forward_hook(save(name)) for name in names
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = (
        load_model(args.root / "checkpoints/bevformer_tiny_epoch_24.pth").cuda().eval()
    )
    print(
        "PASS: strict checkpoint loading:",
        len(model.state_dict()),
        "tensors",
        flush=True,
    )
    reference = make_reference(model, args.root)
    adapted_layers, reference_layers = {}, {}
    handles = []
    if args.diagnose:
        handles += diagnostic_hooks(model, adapted_layers)
        handles += diagnostic_hooks(reference.reference_model, reference_layers)
    report = {
        "scope": "full tensor core, synthetic inputs, PyTorch reference sampler",
        "state_tensors": len(model.state_dict()),
        "cases": [],
    }
    with torch.no_grad():
        for seed, history in ((7, False), (19, True)):
            inputs = tuple(x.cuda() for x in example_inputs(seed=seed, history=history))
            actual = model(*inputs)
            expected = reference(*inputs)
            if args.diagnose:
                for name in adapted_layers:
                    difference = (adapted_layers[name] - reference_layers[name]).abs()
                    print(
                        "LAYER",
                        name,
                        "max_abs_error",
                        difference.max().item(),
                        flush=True,
                    )
            errors = []
            for name, output, wanted in zip(
                ("class_logits", "boxes", "next_bev"), actual, expected
            ):
                torch.testing.assert_close(output, wanted, rtol=3e-4, atol=3e-4)
                errors.append(
                    {
                        "output": name,
                        "max_abs_error": (output - wanted).abs().max().item(),
                    }
                )
            case = {"seed": seed, "history": history, "errors": errors}
            report["cases"].append(case)
            print("PASS: upstream full-forward comparison", case, flush=True)
        for handle in handles:
            handle.remove()
        if not args.skip_export:
            exported = torch.export.export(model, inputs, strict=True)
            print(
                "PASS: strict export",
                len(list(exported.graph.nodes)),
                "nodes",
                flush=True,
            )
            for seed, history in ((31, False), (51, True)):
                alternate = tuple(
                    x.cuda() for x in example_inputs(seed=seed, history=history)
                )
                for actual, expected in zip(
                    exported.module()(*alternate), model(*alternate)
                ):
                    torch.testing.assert_close(actual, expected)
                print(
                    "PASS: exported graph changed inputs/history",
                    seed,
                    history,
                    flush=True,
                )
            report["strict_export_changed_inputs"] = "PASS"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("PASS:", args.report, flush=True)


if __name__ == "__main__":
    main()
