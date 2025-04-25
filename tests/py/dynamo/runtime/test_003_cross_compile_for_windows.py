import os
import platform
import tempfile
import unittest

import pytest
import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase
from torch_tensorrt.dynamo.utils import get_model_device
from torch_tensorrt._utils import check_cross_compile_trt_win_lib

from ..testing_utilities import DECIMALS_OF_AGREEMENT


class TestCrossCompileSaveForWindows(TestCase):
    @unittest.skipIf(
        platform.system() != "Linux" or platform.architecture()[0] != "64bit",
        "Cross compile for windows can only be enabled on linux x86-64 platform",
    )
    @unittest.skipIf(
        not (check_cross_compile_trt_win_lib()),
        "TRT windows lib for cross compile not found",
    )
    @pytest.mark.unit
    def test_cross_compile_for_windows(self):
        class Add(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        model = Add().eval().cuda()
        inputs = [torch.randn(2, 3).cuda(), torch.randn(2, 3).cuda()]
        trt_ep_path = os.path.join(tempfile.gettempdir(), "trt.ep")
        compile_spec = {
            "inputs": inputs,
            "min_block_size": 1,
        }
        try:
            torch_tensorrt.cross_compile_for_windows(
                model, file_path=trt_ep_path, **compile_spec
            )
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

    @unittest.skipIf(
        platform.system() != "Linux" or platform.architecture()[0] != "64bit",
        "Cross compile for windows can only be enabled on linux x86-64 platform",
    )
    @unittest.skipIf(
        not (
            check_cross_compile_trt_win_lib(),
            "TRT windows lib for cross compile not found",
        ),
    )
    @pytest.mark.unit
    def test_dynamo_cross_compile_for_windows(self):
        class Add(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        model = Add().eval().cuda()
        inputs = (torch.randn(2, 3).cuda(), torch.randn(2, 3).cuda())
        trt_ep_path = os.path.join(tempfile.gettempdir(), "trt.ep")
        exp_program = torch.export.export(model, inputs)
        compile_spec = {
            "inputs": inputs,
            "min_block_size": 1,
        }
        try:
            trt_gm = torch_tensorrt.dynamo.cross_compile_for_windows(
                exp_program, **compile_spec
            )
            torch_tensorrt.dynamo.save_cross_compiled_exported_program(
                trt_gm, file_path=trt_ep_path
            )
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

    @unittest.skipIf(
        platform.system() != "Linux" or platform.architecture()[0] != "64bit",
        "Cross compile for windows can only be enabled on linux x86-64 platform",
    )
    @pytest.mark.unit
    def test_dynamo_cross_compile_for_windows_cpu_offload(self):
        class Add(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        model = Add().eval().cuda()
        inputs = (torch.randn(2, 3).cuda(), torch.randn(2, 3).cuda())
        trt_ep_path = os.path.join(tempfile.gettempdir(), "trt.ep")
        exp_program = torch.export.export(model, inputs)
        compile_spec = {
            "inputs": inputs,
            "min_block_size": 1,
            "offload_module_to_cpu": True,
        }
        try:
            trt_gm = torch_tensorrt.dynamo.cross_compile_for_windows(
                exp_program, **compile_spec
            )
            assert get_model_device(trt_gm).type == "cpu"
            torch_tensorrt.dynamo.save_cross_compiled_exported_program(
                trt_gm, file_path=trt_ep_path
            )
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

    @unittest.skipIf(
        platform.system() != "Linux" or platform.architecture()[0] != "64bit",
        "Cross compile for windows can only be enabled on linux x86-64 platform",
    )
    @pytest.mark.unit
    def test_dynamo_cross_compile_for_windows_multiple_output(self):
        class Add(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b), torch.add(a, b)

        model = Add().eval().cuda()
        inputs = (torch.randn(2, 3).cuda(), torch.randn(2, 3).cuda())
        trt_ep_path = os.path.join(tempfile.gettempdir(), "trt.ep")
        exp_program = torch.export.export(model, inputs)
        compile_spec = {
            "inputs": inputs,
            "min_block_size": 1,
        }
        try:
            trt_gm = torch_tensorrt.dynamo.cross_compile_for_windows(
                exp_program, **compile_spec
            )
            torch_tensorrt.dynamo.save_cross_compiled_exported_program(
                trt_gm, file_path=trt_ep_path
            )
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")
