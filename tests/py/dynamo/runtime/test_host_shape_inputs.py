import torch
import torch.nn as nn
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule


class ShapeInputModel(nn.Module):
    """``arange(0, n, 1)`` makes ``n`` a TensorRT shape-tensor input binding."""

    def forward(self, x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        a = torch.arange(0, n, 1, device=x.device).to(torch.float32)
        return (x * 2.0 + a).sum(dim=0, keepdim=True)


class TestHostShapeInput(TestCase):
    """A shape-tensor input is read from host memory, so the C++ runtime accepts it on the
    host without a device round trip. The arange converter tests only exercise a CUDA shape
    input; these cover the host path."""

    def _compile(self):
        model = ShapeInputModel().eval().cuda()
        x = torch.randn(64, device="cuda")
        n = torch.tensor(64, dtype=torch.int64, device="cuda")
        dim = torch.export.Dim("d", min=2, max=256)
        ep = torch.export.export(
            model, (x, n), dynamic_shapes={"x": {0: dim}, "n": None}
        )
        gm = torch_tensorrt.dynamo.compile(
            ep,
            inputs=[x, n],
            min_block_size=1,
            use_python_runtime=False,
            pass_through_build_failures=True,
            assume_dynamic_shape_support=True,
        )
        trt_mod = next(
            m for _, m in gm.named_children() if isinstance(m, TorchTensorRTModule)
        )
        self.assertEqual(
            set(trt_mod.input_binding_names),
            {"x", "_local_scalar_dense"},
            "unexpected engine input bindings; the shape-input mapping below needs updating",
        )
        return model, trt_mod

    def _run_engine(self, trt_mod, x, shape_val, shape_device):
        args = {
            "x": x,
            "_local_scalar_dense": torch.tensor(
                shape_val, dtype=torch.int64, device=shape_device
            ),
        }
        ordered = [args[name] for name in trt_mod.input_binding_names]
        return torch.ops.tensorrt.execute_engine(ordered, trt_mod.engine)[0]

    def test_host_shape_input_matches_eager(self):
        model, trt_mod = self._compile()
        for shape_val in (8, 64, 200):
            x = torch.randn(shape_val, device="cuda")
            eager = model(x, torch.tensor(shape_val, device="cuda"))
            host_out = self._run_engine(trt_mod, x, shape_val, "cpu")
            torch.testing.assert_close(host_out, eager, rtol=1e-3, atol=1e-3)

    def test_host_and_device_shape_input_agree(self):
        model, trt_mod = self._compile()
        x = torch.randn(64, device="cuda")
        host_out = self._run_engine(trt_mod, x, 64, "cpu")
        device_out = self._run_engine(trt_mod, x, 64, "cuda")
        torch.testing.assert_close(host_out, device_out, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    run_tests()
