import pytest
import torch
import torch_tensorrt
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch_tensorrt.dynamo._exporter import transform


class HostBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 8))

    def forward(self, x):
        return x.to(self.weight.device) @ self.weight


class HybridHostModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.host = HostBlock()

    def forward(self, x):
        y = x * 2.0 + 1.0
        host = self.host(y.to("cpu"))
        return host.to(y.device) + y


@pytest.mark.unit
def test_compile_preserves_host_fallback_parameter_placement():
    model = HybridHostModel().eval()
    x = torch.randn(8, 8, device="cuda")
    expected = model(x)

    exported = torch.export.export(model, (x,))
    compiled = torch_tensorrt.dynamo.compile(
        exported,
        inputs=(x,),
        min_block_size=1,
        pass_through_build_failures=True,
        torch_executed_ops={
            "torch.ops.aten.matmul.default",
            "torch.ops.aten._to_copy.default",
        },
    )

    fallback_devices = [
        parameter.device
        for name, child in compiled.named_children()
        if "_run_on_acc" not in name
        for parameter in child.parameters()
    ]
    assert fallback_devices
    assert all(device.type == "cpu" for device in fallback_devices)
    torch.testing.assert_close(compiled(x), expected)

    # Re-export exercises FakeTensor propagation through the hybrid boundary.
    torch_tensorrt.dynamo.export(compiled, arg_inputs=(x,))


class HostOutput(nn.Module):
    def forward(self, x):
        return (x * 2.0).to(device="cpu", dtype=torch.int32)


@pytest.mark.unit
def test_device_changing_to_copy_stays_outside_engine():
    model = HostOutput().eval().cuda()
    x = torch.randn(16, device="cuda")
    expected = model(x)

    exported = torch.export.export(model, (x,))
    compiled = torch_tensorrt.dynamo.compile(
        exported,
        inputs=(x,),
        min_block_size=1,
        pass_through_build_failures=True,
    )

    actual = compiled(x)
    assert actual.device.type == "cpu"
    torch.testing.assert_close(actual, expected)

    inlined = transform(compiled)
    inlined.recompile()
    with FakeTensorMode(shape_env=ShapeEnv()):
        fake_output = inlined(torch.empty(16, device="cuda"))
    assert fake_output.device.type == "cpu"


class HostComputeAfterCopy(nn.Module):
    def forward(self, x):
        host = (x * 2.0).to("cpu")
        return host + 1.0


@pytest.mark.unit
@pytest.mark.parametrize("use_fast_partitioner", [True, False])
def test_cpu_compute_after_device_copy_stays_outside_engine(use_fast_partitioner):
    model = HostComputeAfterCopy().eval().cuda()
    x = torch.randn(16, device="cuda")
    expected = model(x)

    exported = torch.export.export(model, (x,))
    compiled = torch_tensorrt.dynamo.compile(
        exported,
        inputs=(x,),
        min_block_size=1,
        pass_through_build_failures=True,
        use_fast_partitioner=use_fast_partitioner,
    )

    assert any("_run_on_acc" in name for name, _ in compiled.named_children())
    actual = compiled(x)
    assert actual.device.type == "cpu"
    torch.testing.assert_close(actual, expected)
