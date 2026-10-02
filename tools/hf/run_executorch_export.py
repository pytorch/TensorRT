#!/usr/bin/env python3
"""Compile a supported Edge model and package its ExecuTorch programs."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch_tensorrt  # noqa: F401
from exporters import EdgeConfig
from exporters.executorch.exporter import EdgeExecuTorchExporter
from exporters.measure import print_bench
from exporters.plugin.plugin_utils import load_plugins_for_trt

PreparedExport = tuple[nn.Module, MutableMapping[str, Any], EdgeConfig, str]
ExportPreparer = Callable[
    [argparse.Namespace, torch.device, torch.dtype],
    PreparedExport,
]

# Add a family here once all of its component model types have packagers.
EXPORT_PREPARERS = {
    "pi05": "exporters.models.pi05.export:prepare_export",
}


def get_export_preparer(model_type: str) -> ExportPreparer:
    module_name, function_name = EXPORT_PREPARERS[model_type].split(":")
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_type", choices=EXPORT_PREPARERS)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/pi05_executorch"),
    )
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16"),
        default="float16",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Edge ExecuTorch export requires a CUDA device")

    load_plugins_for_trt()
    dtype = getattr(torch, args.dtype)

    # The existing family preparers accept engine_dir. The ExecuTorch exporter
    # owns its engine directory under output_dir, so provide that path here.
    args.engine_dir = str(args.output_dir / "engines")
    prepare_export = get_export_preparer(args.model_type)
    model, sample_inputs, config, _ = prepare_export(args, device, dtype)

    device_id = device.index
    if device_id is None:
        device_id = torch.cuda.current_device()

    exporter = EdgeExecuTorchExporter()
    programs = exporter.export(
        model,
        sample_inputs,
        config,
        output_dir=args.output_dir,
        device_id=device_id,
    )

    print("engines:", exporter.engines)
    print("programs:", programs)
    print("manifest:", exporter.manifest_path)
    print_bench(exporter.bench)


if __name__ == "__main__":
    main()
