#!/usr/bin/env python3
"""Run an EdgeExporter smoke test for a supported model family."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable, MutableMapping
from typing import Any

import torch
import torch.nn as nn
import torch_tensorrt  # noqa: F401
from exporters import EdgeConfig, EdgeExporter
from exporters.measure import print_bench
from exporters.plugin.plugin_utils import load_plugins_for_trt

PreparedExport = tuple[nn.Module, MutableMapping[str, Any], EdgeConfig, str]
ExportPreparer = Callable[
    [argparse.Namespace, torch.device, torch.dtype], PreparedExport
]

EXPORT_PREPARERS = {
    "alpamayo": "exporters.models.alpamayo.export:prepare_export",
    "groot": "exporters.models.groot.export:prepare_export",
    "pi05": "exporters.models.pi05.export:prepare_export",
    "nemotron": "exporters.models.nemotron.export:prepare_export",
    "nanbeige": "exporters.models.nanbeige.export:prepare_export",
    "kimi": "exporters.models.kimi.export:prepare_export",
}


def get_export_preparer(model_type: str) -> ExportPreparer:
    module_name, function_name = EXPORT_PREPARERS[model_type].split(":")
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_type", choices=EXPORT_PREPARERS)
    parser.add_argument("--checkpoint")
    parser.add_argument("--prompt", default="Hello, how are you?")
    parser.add_argument("--engine-dir")
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument(
        "--clip-id",
        help="PhysicalAI clip ID used to prepare Alpamayo sample inputs.",
    )
    parser.add_argument(
        "--t0-us",
        type=int,
        default=5_100_000,
        help="Timestamp within the Alpamayo sample clip (default: 5100000).",
    )
    parser.add_argument("--device")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16"),
        default="float16",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_plugins_for_trt()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = getattr(torch, args.dtype)
    prepare_export = get_export_preparer(args.model_type)
    model, sample_inputs, config, output_name = prepare_export(args, device, dtype)

    exporter = EdgeExporter()
    program = exporter.export(model, sample_inputs, config=config)

    print("engines:", exporter.engines)
    if program is None:
        print_bench(exporter.bench)
        return

    print("runtime keys:", sorted(exporter.sample))
    with torch.no_grad():
        result = program.module()(**exporter.sample)

    output = result[0] if isinstance(result, (tuple, list)) else result
    print(
        output_name,
        tuple(output.shape),
        "mean",
        float(output.float().mean()),
    )
    print_bench(exporter.bench)


if __name__ == "__main__":
    main()
