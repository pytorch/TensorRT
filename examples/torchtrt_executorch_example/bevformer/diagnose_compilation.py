# SPDX-License-Identifier: Apache-2.0
"""Isolate numerical changes in TensorRT conversion before exporting a .pte."""

import argparse
import json
from pathlib import Path

import torch
import torch_tensorrt
from model import example_inputs, load_model
from torch import nn


class GeometryCall(nn.Module):
    def __init__(self, geometry, method):
        super().__init__()
        self.geometry, self.method = geometry, method

    def forward(self, *args):
        return getattr(self.geometry, self.method)(*args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--stage", action="append", help="Limit checks to named stages")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_model(args.root / "checkpoints/bevformer_tiny_epoch_24.pth").cuda()
    inputs = tuple(t.cuda() for t in example_inputs(seed=19, history=True))
    names = {
        "backbone": "img_backbone",
        "temporal": "pts_bbox_head.transformer.encoder.layers.0.attentions.0",
        "spatial": "pts_bbox_head.transformer.encoder.layers.0.attentions.1",
        "query_attention": "pts_bbox_head.transformer.decoder.layers.0.attentions.0",
        "decoder_cross": "pts_bbox_head.transformer.decoder.layers.0.attentions.1",
    }
    captured = {}

    def capture(name):
        def hook(_module, values):
            captured[name] = tuple(v.detach().clone() for v in values)

        return hook

    handles = [
        model.get_submodule(path).register_forward_pre_hook(capture(name))
        for name, path in names.items()
    ]
    with torch.no_grad():
        model(*inputs)
        for handle in handles:
            handle.remove()
        stages = {
            "projection": (GeometryCall(model.geometry, "project"), (inputs[1],)),
            "rotation": (
                GeometryCall(model.geometry, "rotate"),
                (inputs[3], inputs[2][:, -1]),
            ),
            **{
                name: (model.get_submodule(path), captured[name])
                for name, path in names.items()
            },
        }
        results = {}
        for name, (module, values) in stages.items():
            if args.stage and name not in args.stage:
                continue
            print("CHECK:", name, flush=True)
            module.eval()
            expected = module(*values)
            exported = torch.export.export(module, values, strict=True)
            try:
                compiled = torch_tensorrt.dynamo.compile(
                    exported,
                    arg_inputs=values,
                    min_block_size=1,
                    disable_tf32=True,
                    enabled_precisions={torch.float32},
                    workspace_size=2 << 30,
                    pass_through_build_failures=True,
                )
                actual = compiled(*values)
                if isinstance(expected, torch.Tensor):
                    expected, actual = (expected,), (actual,)
                errors = []
                passed = True
                for output, wanted in zip(actual, expected):
                    errors.append((output - wanted).abs().max().item())
                    try:
                        torch.testing.assert_close(output, wanted, rtol=3e-4, atol=3e-4)
                    except AssertionError:
                        passed = False
                results[name] = {"pass": passed, "max_abs_errors": errors}
            except Exception as error:
                results[name] = {"pass": False, "error": repr(error)}
            print("RESULT:", name, results[name], flush=True)
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
