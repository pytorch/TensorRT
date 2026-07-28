"""Subprocess helper: run TRT operations in a Python-only environment.

This script temporarily hides the Torch-TensorRT C++ shared libraries so that
``torch_tensorrt`` imports in Python-only mode.

Modes:
    load   — Load a .pt2 artifact, run inference, save output.
    save   — Compile a model and save a .pt2 artifact.

Usage (called by test_cross_runtime_serde.py, not directly):
    python _cross_runtime_load_helper.py load \
        --artifact <path.pt2> --input <input.pt> --output <output.pt>

    python _cross_runtime_load_helper.py save \
        --model-state <model_state.pt> --input <input.pt> --artifact <path.pt2>
"""

from __future__ import annotations

import argparse
import glob
import os
import sys


def _build_small_conv_model(torch):
    class SmallConvModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, bias=True)
            self.relu = torch.nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.relu(self.conv(x))

    return SmallConvModel()


def _normalize_outputs(result):
    if isinstance(result, tuple):
        return result
    if isinstance(result, list):
        return tuple(result)
    return (result,)


def _save_inference_output(ep, inp, output_path: str) -> None:
    import torch

    with torch.no_grad():
        result = ep.module()(inp)
    torch.save(_normalize_outputs(result), output_path)


def _assert_python_runtime_only(torchtrt) -> None:
    assert (
        not torchtrt.ENABLED_FEATURES.torch_tensorrt_runtime
    ), "C++ runtime should be disabled"


def _compile_spec(inp, torchtrt) -> dict:
    import torch

    return {
        "inputs": [
            torchtrt.Input(inp.shape, dtype=torch.float, format=torch.contiguous_format)
        ],
        "ir": "dynamo",
        "min_block_size": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }


def _hide_so_files(pkg_dir: str) -> list[tuple[str, str]]:
    """Rename .so/.dll files in pkg_dir/lib so _features.py sees them as absent."""
    lib_dir = os.path.join(pkg_dir, "lib")
    if not os.path.isdir(lib_dir):
        return []
    # Linux/macOS uses `libtorchtrt*` (the `lib` prefix); Windows uses
    # `torchtrt*.dll` / `torchtrt*.lib` (no prefix). See ``_features.py``
    # for the matching read side.
    patterns = (
        ["torchtrt*.dll", "torchtrt*.lib"]
        if sys.platform.startswith("win")
        else ["libtorchtrt*"]
    )
    moved: list[tuple[str, str]] = []
    for pattern in patterns:
        for path in glob.glob(os.path.join(lib_dir, pattern)):
            bak = path + ".bak"
            os.rename(path, bak)
            moved.append((bak, path))
    return moved


def _restore_so_files(moved: list[tuple[str, str]]) -> None:
    for bak, orig in moved:
        if os.path.exists(bak):
            os.rename(bak, orig)


def _do_load(args: argparse.Namespace) -> None:
    """Load a pre-saved .pt2, run inference, save output."""
    import torch
    import torch_tensorrt as torchtrt

    _assert_python_runtime_only(torchtrt)

    ep = torchtrt.load(args.artifact)
    inp = torch.load(args.input, weights_only=True)
    _save_inference_output(ep, inp, args.output)


def _do_save(args: argparse.Namespace) -> None:
    """Compile and save a .pt2 artifact in Python-only mode."""
    import torch
    import torch_tensorrt as torchtrt

    _assert_python_runtime_only(torchtrt)

    model = _build_small_conv_model(torch).eval().cuda()
    model.load_state_dict(torch.load(args.model_state, weights_only=True))
    inp = torch.load(args.input, weights_only=True)
    compile_spec = _compile_spec(inp, torchtrt)
    exp_program = torchtrt.dynamo.trace(model, **compile_spec)
    trt_module = torchtrt.dynamo.compile(exp_program, **compile_spec)
    torchtrt.save(trt_module, args.artifact, retrace=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    p_load = sub.add_parser("load")
    p_load.add_argument("--artifact", required=True)
    p_load.add_argument("--input", required=True)
    p_load.add_argument("--output", required=True)

    p_save = sub.add_parser("save")
    p_save.add_argument("--model-state", required=True)
    p_save.add_argument("--input", required=True)
    p_save.add_argument("--artifact", required=True)

    args = parser.parse_args()

    import importlib.util

    spec = importlib.util.find_spec("torch_tensorrt")
    assert spec and spec.origin
    pkg_dir = os.path.dirname(spec.origin)

    handlers = {"load": _do_load, "save": _do_save}
    moved = _hide_so_files(pkg_dir)
    try:
        handlers[args.mode](args)
    finally:
        _restore_so_files(moved)


if __name__ == "__main__":
    main()
