# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Time constant_fold with and without the large aliased view execute-skip.

Builds a stack of Linear layers (each exports as permute(weight) -> mm/addmm) and
times torch_tensorrt's constant_fold in two configurations:

  skip     skip_large_aliased_view_fold enabled (this PR)
  no-skip  views are executed as before; the install-time skip (#4531) still applies

Reports median wall time over --repeats runs and how many permute/mul ops remain.
`--scaled` adds a foldable `weight.T * 2` consumer to show that downstream folds
are preserved.

    python tools/perf/constant_fold_skip_timing.py --layers 24 --features 2048
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes import constant_folding as cf


class _Stack(torch.nn.Module):
    def __init__(self, layers: int, features: int, scaled: bool) -> None:
        super().__init__()
        self.scaled = scaled
        self.linears = torch.nn.ModuleList(
            torch.nn.Linear(features, features) for _ in range(layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.linears:
            if self.scaled:
                x = x @ (layer.weight.permute(1, 0) * 2.0) + layer.bias
            else:
                x = layer(x)
        return x


def _count(gm: torch.fx.GraphModule, target: object) -> int:
    return sum(1 for n in gm.graph.nodes if n.target is target)


def _run(args: argparse.Namespace, skip: bool) -> tuple[float, int, int]:
    model = _Stack(args.layers, args.features, args.scaled).eval()
    ep = torch.export.export(
        model, (torch.randn(4, args.features),)
    ).run_decompositions()
    gm = ep.module()
    original = cf.skip_large_aliased_view_fold
    if not skip:
        cf.skip_large_aliased_view_fold = lambda gm, node: False  # type: ignore
    try:
        start = time.perf_counter()
        folded = cf.constant_fold(gm, CompilationSettings())
        elapsed = time.perf_counter() - start
    finally:
        cf.skip_large_aliased_view_fold = original  # type: ignore
    return (
        elapsed,
        _count(folded, torch.ops.aten.permute.default)
        + _count(folded, torch.ops.aten.t.default),
        _count(folded, torch.ops.aten.mul.Tensor),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--features", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--scaled", action="store_true")
    parser.add_argument("--reverse", action="store_true", help="run no-skip first")
    args = parser.parse_args()

    for skip in ((False, True) if args.reverse else (True, False)):
        runs = [_run(args, skip) for _ in range(args.repeats)]
        print(
            f"{'skip' if skip else 'no-skip':8s} "
            f"median {statistics.median(r[0] for r in runs) * 1e3:9.1f} ms  "
            f"permutes left {runs[-1][1]:3d}  muls left {runs[-1][2]:3d}"
        )


if __name__ == "__main__":
    main()
