# SPDX-License-Identifier: BSD-3-Clause
"""CPU regressions for output metadata after device-copy insertion."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Requires PyTorch; no GPU is needed")
class CallerIOTests(unittest.TestCase):
    def test_cpu_outputs_replace_stale_gpu_specs_without_changing_intermediates(self):
        with patch.object(
            sys, "path", [str(Path(__file__).resolve().parents[4]), *sys.path]
        ):
            from examples.torchtrt_executorch_example.native_validation.optimizations.caller_owned_io import (
                refresh_output_specs,
            )

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        gpu = graph.call_function(torch.neg, (x,))
        cpu = graph.call_function(torch.clone, (gpu,))
        gpu_spec, cpu_spec = object(), object()
        gpu.meta["spec"], cpu.meta["spec"] = gpu_spec, cpu_spec
        output = graph.output((cpu, [cpu, None]))
        output.meta["spec"] = (gpu_spec, [gpu_spec, None])
        module = torch.fx.GraphModule({}, graph)
        before = module(torch.ones(2))
        refresh_output_specs(module)
        self.assertIs(output.meta["spec"][0], cpu_spec)
        self.assertIs(output.meta["spec"][1][0], cpu_spec)
        self.assertIsNone(output.meta["spec"][1][1])
        self.assertIs(gpu.meta["spec"], gpu_spec)
        torch.testing.assert_close(module(torch.ones(2)), before)
        refresh_output_specs(module)  # Already refreshed metadata stays valid.
        self.assertIs(output.meta["spec"][0], cpu_spec)

    def test_reject_planned_public_io_but_allow_planned_intermediates(self):
        with patch.object(
            sys, "path", [str(Path(__file__).resolve().parents[4]), *sys.path]
        ):
            from examples.torchtrt_executorch_example.native_validation.optimizations.caller_owned_io import (
                validate_program,
            )

        values = [
            NS(val=NS(allocation_info=None)),
            NS(val=NS(allocation_info=None)),
            NS(val=NS(allocation_info=object())),
        ]
        program = NS(execution_plan=[NS(inputs=[0], outputs=[1], values=values)])
        validate_program(program)
        for index in (0, 1):
            values[index].val.allocation_info = object()
            with self.assertRaisesRegex(ValueError, "caller-owned"):
                validate_program(program)
            values[index].val.allocation_info = None


if __name__ == "__main__":
    unittest.main()
