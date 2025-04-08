# type: ignore
import importlib
import platform
import unittest
from importlib import metadata

import pytest
import torch
import torch_tensorrt as torchtrt

from packaging.version import Version

assertions = unittest.TestCase()


@unittest.skipIf(
    torch.cuda.get_device_capability() < (8, 9),
    "FP8 quantization requires compute capability 8.9 or later",
)
@unittest.skipIf(
    not importlib.util.find_spec("modelopt"),
    "ModelOpt is required to run this test",
)
@pytest.mark.unit
def test_base_fp8(ir):
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.utils import export_torch_mode

    class SimpleNetwork(torch.nn.Module):
        def __init__(self):
            super(SimpleNetwork, self).__init__()
            self.linear1 = torch.nn.Linear(in_features=10, out_features=5)
            self.linear2 = torch.nn.Linear(in_features=5, out_features=1)

        def forward(self, x):
            x = self.linear1(x)
            x = torch.nn.ReLU()(x)
            x = self.linear2(x)
            return x

    def calibrate_loop(model):
        """Simple calibration function for testing."""
        model(input_tensor)

    input_tensor = torch.randn(1, 10).cuda()
    model = SimpleNetwork().eval().cuda()

    quant_cfg = mtq.FP8_DEFAULT_CFG
    mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    # model has FP8 qdq nodes at this point
    output_pyt = model(input_tensor)

    with torch.no_grad():
        with export_torch_mode():
            exp_program = torch.export.export(model, (input_tensor,), strict=False)
            trt_model = torchtrt.dynamo.compile(
                exp_program,
                inputs=[input_tensor],
                enabled_precisions={torch.float8_e4m3fn},
                min_block_size=1,
                debug=True,
                cache_built_engines=False,
                reuse_cached_engines=False,
            )
            outputs_trt = trt_model(input_tensor)
            assert torch.allclose(output_pyt, outputs_trt, rtol=5e-3, atol=1e-2)


@unittest.skipIf(
    platform.system() != "Linux"
    or not importlib.util.find_spec("modelopt")
    or Version(metadata.version("nvidia-modelopt")) < Version("0.17.0"),
    "modelopt 0.17.0 or later is required, Int8 quantization is supported in modelopt since 0.17.0 or later for linux",
)
@pytest.mark.unit
def test_base_int8(ir):
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.utils import export_torch_mode

    class SimpleNetwork(torch.nn.Module):
        def __init__(self):
            super(SimpleNetwork, self).__init__()
            self.linear1 = torch.nn.Linear(in_features=10, out_features=5)
            self.linear2 = torch.nn.Linear(in_features=5, out_features=1)

        def forward(self, x):
            x = self.linear1(x)
            x = torch.nn.ReLU()(x)
            x = self.linear2(x)
            return x

    def calibrate_loop(model):
        """Simple calibration function for testing."""
        model(input_tensor)

    input_tensor = torch.randn(1, 10).cuda()
    model = SimpleNetwork().eval().cuda()

    quant_cfg = mtq.INT8_DEFAULT_CFG
    mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    # model has INT8 qdq nodes at this point
    output_pyt = model(input_tensor)

    with torch.no_grad():
        with export_torch_mode():
            exp_program = torch.export.export(model, (input_tensor,))
            trt_model = torchtrt.dynamo.compile(
                exp_program,
                inputs=[input_tensor],
                enabled_precisions={torch.int8},
                min_block_size=1,
                debug=True,
                cache_built_engines=False,
                reuse_cached_engines=False,
                truncate_double=True,
            )
            outputs_trt = trt_model(input_tensor)
            assert torch.allclose(output_pyt, outputs_trt, rtol=5e-3, atol=1e-2)
