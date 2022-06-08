import io
import unittest

import torch
import torch.fx
from torch import nn
from torch.package import PackageImporter
from torch_tensorrt.fx.tools.model_packager import (
    generate_standalone_repro,
    ModelPackager,
)


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Module()
        self.b = torch.nn.Module()
        self.a.weights = torch.nn.Parameter(torch.randn(1, 2))
        self.b.weights = torch.nn.Parameter(
            torch.randn(
                1,
            )
        )

    def forward(self, x):
        return x + self.a.weights + self.b.weights


class ModelPackagerTest(unittest.TestCase):
    def test_text_repro_gen(self):
        model = torch.fx.symbolic_trace(TestModel().eval())
        inputs = [torch.randn(1)]
        _ = model(*inputs)

        string_io = io.StringIO()
        generate_standalone_repro(model, string_io, "\n# hello")
        string_io.seek(0)
        exec(string_io.read())
        exported_model = locals()["ExportedModule"]()
        _ = exported_model(*inputs)

    def test_package_model(self):
        model = torch.fx.symbolic_trace(TestModel().eval())
        inputs = [torch.randn(1)]
        _ = model(*inputs)
        bytesIO = io.BytesIO()
        ModelPackager.package_model(model, inputs, bytesIO)
        bytesIO.seek(0)
        pi = PackageImporter(bytesIO)
        reload_model = pi.load_pickle("repro", "model")
        reload_inputs = pi.load_pickle("repro", "inputs")

        torch.testing.assert_allclose(model(*inputs), reload_model(*reload_inputs))
        keys = dict(reload_model.named_children()).keys()
        self.assertEqual(keys, {"_holder"})
