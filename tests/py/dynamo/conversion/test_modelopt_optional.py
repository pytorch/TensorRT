# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
import sys
import textwrap
import unittest


class TestModelOptOptional(unittest.TestCase):
    def test_unquantized_model_does_not_import_modelopt(self):
        # Use a fresh interpreter to cover converter registration during import,
        # even when another test has already imported ModelOpt.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent("""
                    import importlib.abc
                    import sys

                    attempts = []

                    class BlockModelOpt(importlib.abc.MetaPathFinder):
                        def find_spec(self, fullname, path=None, target=None):
                            if fullname == "modelopt" or fullname.startswith("modelopt."):
                                attempts.append(fullname)
                                raise ImportError("ModelOpt is unavailable")

                    sys.meta_path.insert(0, BlockModelOpt())
                    import torch
                    import torch_tensorrt
                    from torch_tensorrt.dynamo._settings import CompilationSettings
                    from torch_tensorrt.dynamo.conversion import DYNAMO_CONVERTERS
                    from torch_tensorrt.dynamo.lowering.passes.constant_folding import (
                        _TorchTensorRTConstantFolder,
                    )
                    from torch_tensorrt.dynamo.runtime._MutableTorchTensorRTModule import (
                        _is_modelopt_quantized,
                    )
                    from torch_tensorrt.dynamo.utils import is_quantized_by_modelopt

                    model = torch.nn.Linear(4, 4)
                    assert not is_quantized_by_modelopt(model)
                    assert not _is_modelopt_quantized(model)
                    gm = torch.fx.symbolic_trace(model)
                    _TorchTensorRTConstantFolder(gm)
                    DYNAMO_CONVERTERS.set_compilation_settings(CompilationSettings())
                    assert not attempts, attempts

                    # Quantizer subclasses are detected without importing ModelOpt.
                    TensorQuantizer = type(
                        "TensorQuantizer", (torch.nn.Module,),
                        {"__module__": "modelopt.torch.quantization.nn"},
                    )
                    class CustomQuantizer(TensorQuantizer):
                        pass
                    assert is_quantized_by_modelopt(CustomQuantizer())
                    assert _is_modelopt_quantized(CustomQuantizer())

                    # Operators loaded after Torch-TensorRT still get converters.
                    lib = torch.library.Library("tensorrt", "FRAGMENT")
                    lib.define("quantize_op(Tensor x, Tensor amax, int bits, int axis) -> Tensor")
                    op = torch.ops.tensorrt.quantize_op.default
                    DYNAMO_CONVERTERS.set_compilation_settings(CompilationSettings())
                    assert op in DYNAMO_CONVERTERS
                    DYNAMO_CONVERTERS.set_compilation_settings(CompilationSettings())
                    from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
                        DYNAMO_ATEN_CONVERTERS,
                    )
                    assert len(DYNAMO_ATEN_CONVERTERS[op]) == 1
                    assert op in _TorchTensorRTConstantFolder(gm).quantization_ops
                    assert not attempts, attempts
                    """),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
