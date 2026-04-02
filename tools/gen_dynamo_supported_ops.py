"""Generate docsrc/indices/supported_ops.rst from Dynamo opset coverage data.

Usage:
    python tools/gen_dynamo_supported_ops.py

Requires the opset coverage JSON files to exist in /tmp/ (run
``python py/torch_tensorrt/dynamo/tools/opset_coverage.py`` first).
"""

import json
import os
import sys
from pathlib import Path


def load(path):
    with open(path) as f:
        return json.load(f)


def fix_schema(key: str, schema: str) -> str:
    """Reconstruct a display-ready signature from a key + raw schema string.

    The opset_coverage tool stores schemas as e.g.
      prims.(Tensor inp, ...) -> Tensor   (name missing)
      _operator.                           (no signature at all)
    For aten ops the schema is already complete.
    """
    prefix = key.split(".")[0]  # "aten", "prims", "_operator"
    if prefix == "_operator":
        return key  # no schema available, just show the qualified name
    if prefix == "prims":
        # schema = "prims.(args) -> ret" — insert the op name after the dot
        op_name = key[len("prims.") :]  # e.g. "sum"
        return f"prims.{op_name}" + schema[len("prims.") :]
    return schema  # aten schemas are already complete


def ops_by_status(data, status):
    entries = []
    for key, info in data["support_status"].items():
        if info["status"] == status:
            entries.append(fix_schema(key, info["schema"]))
    return sorted(entries)


def section(title, underline_char, items, description):
    lines = [title, underline_char * len(title), ""]
    lines.append(description)
    lines.append("")
    for s in items:
        lines.append(f"- {s}")
    lines.append("")
    return lines


def generate(aten_path, prims_path, py_path, out_path):
    aten = load(aten_path)
    prims = load(prims_path)
    py_ops = load(py_path)

    lines = [
        "",
        ".. _supported_ops:",
        "",
        "=================================",
        "Operators Supported",
        "=================================",
        "",
        ".. note::",
        "",
        "   This page reflects operator coverage for the **Dynamo** "
        "(``torch_tensorrt.dynamo``) frontend.",
        "   Operators marked *Converted* have a native Dynamo converter.",
        "   Operators marked *Lowered* are handled via ATen decompositions "
        "before reaching TensorRT.",
        "",
    ]

    aten_converted = ops_by_status(aten, "CONVERTED")
    aten_lowered = ops_by_status(aten, "LOWERED")
    py_converted = ops_by_status(py_ops, "CONVERTED")
    prims_converted = ops_by_status(prims, "CONVERTED")
    prims_lowered = ops_by_status(prims, "LOWERED")

    lines += section(
        "ATen Core Ops — Converted",
        "-",
        aten_converted,
        f"*{len(aten_converted)} operators with native Dynamo converters.*",
    )

    lines += section(
        "ATen Core Ops — Lowered via Decomposition",
        "-",
        aten_lowered,
        f"*{len(aten_lowered)} operators decomposed into supported primitives "
        "before TensorRT compilation.*",
    )

    lines += section(
        "Python Builtin Ops — Converted",
        "-",
        py_converted,
        f"*{len(py_converted)} Python-level operators supported.*",
    )

    lines += section(
        "Prims Ops — Converted",
        "-",
        prims_converted,
        f"*{len(prims_converted)} prims operators with native Dynamo converters.*",
    )

    lines += section(
        "Prims Ops — Lowered via Decomposition",
        "-",
        prims_lowered,
        f"*{len(prims_lowered)} prims operators decomposed into supported primitives.*",
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Written to {out_path}")
    print(f"  ATen converted: {len(aten_converted)}, lowered: {len(aten_lowered)}")
    print(f"  Python builtins converted: {len(py_converted)}")
    print(f"  Prims converted: {len(prims_converted)}, lowered: {len(prims_lowered)}")


if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent
    generate(
        aten_path="/tmp/aten_coverage_status.json",
        prims_path="/tmp/prim_coverage_status.json",
        py_path="/tmp/py_overload_coverage_status.json",
        out_path=repo_root / "docsrc/indices/supported_ops.rst",
    )
