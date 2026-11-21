# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import importlib.machinery
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch_tensorrt.dynamo.utils import is_quantized_by_modelopt

_QUANT_NN = "modelopt.torch.quantization.nn"


def _drop_modelopt_modules():
    for name in [n for n in sys.modules if n.split(".")[0] == "modelopt"]:
        del sys.modules[name]


class TestIsQuantizedByModelOpt(unittest.TestCase):
    def test_partly_imported_modelopt_is_not_imported_again(self):
        # Like ModelOpt on a PyTorch without TorchScript: its nn module raises AttributeError,
        # not ImportError, and the utils module that loaded before it stays in sys.modules.
        with (
            tempfile.TemporaryDirectory() as root,
            mock.patch.dict(sys.modules),
            mock.patch.object(sys, "path", [root, *sys.path]),
        ):
            _drop_modelopt_modules()
            quantization = Path(root, "modelopt", "torch", "quantization")
            quantization.mkdir(parents=True)
            Path(root, "modelopt", "__init__.py").write_text("")
            Path(root, "modelopt", "torch", "__init__.py").write_text("")
            Path(quantization, "__init__.py").write_text("from . import utils, nn\n")
            Path(quantization, "utils.py").write_text(
                "def is_quantized(module):\n"
                "    from .nn import TensorQuantizer\n"
                "    return any(isinstance(m, TensorQuantizer) for m in module.modules())\n"
            )
            imported = Path(root, "imported")
            Path(quantization, "nn.py").write_text(
                f"open({str(imported)!r}, 'w').close()\n"
                "raise AttributeError('torch.jit.script is missing')\n"
            )
            with self.assertRaises(AttributeError):
                importlib.import_module("modelopt.torch.quantization")
            self.assertIn("modelopt.torch.quantization.utils", sys.modules)
            imported.unlink()

            self.assertFalse(is_quantized_by_modelopt(torch.nn.Linear(2, 2)))
            self.assertFalse(imported.exists())

    def test_blocked_modelopt_is_not_imported(self):
        modelopt = types.ModuleType("modelopt")
        modelopt.__spec__ = importlib.machinery.ModuleSpec(
            "modelopt", None, is_package=True
        )
        with mock.patch.dict(sys.modules):
            _drop_modelopt_modules()
            sys.modules["modelopt"] = modelopt
            for name in ("modelopt.torch.quantization", _QUANT_NN):
                sys.modules[name] = None

            self.assertFalse(is_quantized_by_modelopt(torch.nn.Linear(2, 2)))

    def test_modelopt_still_importing_gives_an_answer(self):
        # Another thread is still running nn.py, so TensorQuantizer is not defined yet.
        with mock.patch.dict(sys.modules, {_QUANT_NN: types.ModuleType(_QUANT_NN)}):
            self.assertFalse(is_quantized_by_modelopt(torch.nn.Linear(2, 2)))

    def test_loaded_tensor_quantizer_is_found(self):
        class TensorQuantizer(torch.nn.Module):
            pass

        quant_nn = types.ModuleType(_QUANT_NN)
        quant_nn.TensorQuantizer = TensorQuantizer
        with mock.patch.dict(sys.modules, {_QUANT_NN: quant_nn}):
            quantized = torch.nn.Sequential(torch.nn.Linear(2, 2), TensorQuantizer())

            self.assertTrue(is_quantized_by_modelopt(quantized))
            self.assertFalse(is_quantized_by_modelopt(torch.nn.Linear(2, 2)))


if __name__ == "__main__":
    unittest.main()
