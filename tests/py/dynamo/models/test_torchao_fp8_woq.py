# type: ignore
import importlib
import unittest

import pytest
import torch
import torch_tensorrt as torchtrt
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity

assertions = unittest.TestCase()


def _has_fp8_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def _promote_float8_woq(model: torch.nn.Module) -> torch.nn.Module:
    from torchao.quantization import dequantize_affine
    from torchao.quantization.quantize_.workflows import Float8Tensor

    class Float8TensorNonDecomposed(Float8Tensor):
        def dequantize(self, output_dtype=None):
            if output_dtype is None:
                output_dtype = torch.bfloat16
            return dequantize_affine(
                self.qdata,
                self.block_size,
                self.scale,
                None,
                self.qdata.dtype,
                output_dtype=output_dtype,
            )

    for param in model.parameters():
        if isinstance(param, Float8Tensor):
            param.__class__ = Float8TensorNonDecomposed
            param.requires_grad_(False)
    return model


@pytest.mark.unit
@unittest.skipIf(importlib.util.find_spec("torchao") is None, "torchao not installed")
@unittest.skipIf(not _has_fp8_gpu(), "FP8 GPU (compute capability >= 9.0) is required")
def test_linear_fp8_woq():
    from torch._inductor.constant_folding import (
        _dont_constant_fold,
        add_dont_constant_fold,
    )
    from torchao.quantization import Float8WeightOnlyConfig, quantize_

    class LinearModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(64, 128)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    model = LinearModel().eval().to(dtype=torch.bfloat16, device="cuda")
    example_input = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda")
    quantize_(model, Float8WeightOnlyConfig())
    model = _promote_float8_woq(model)

    op = torch.ops.torchao.dequantize_affine.default
    add_dont_constant_fold(op)
    try:
        exp_program = torch.export.export(model, (example_input,), strict=True)
    finally:
        if op in _dont_constant_fold:
            _dont_constant_fold.remove(op)

    dq_nodes = [
        n
        for n in exp_program.graph.nodes
        if n.target == torch.ops.torchao.dequantize_affine.default
    ]
    assertions.assertTrue(
        len(dq_nodes) >= 1,
        msg="Exported graph is missing torchao.dequantize_affine",
    )

    trt_mod = torchtrt.dynamo.compile(
        exp_program,
        inputs=[example_input],
        min_block_size=1,
        use_explicit_typing=True,
        require_full_compilation=True,
        cache_built_engines=False,
        reuse_cached_engines=False,
    )

    with torch.no_grad():
        eager_out = model(example_input)
        trt_out = trt_mod(example_input)
    if isinstance(trt_out, (list, tuple)):
        trt_out = trt_out[0]
    cos_sim = cosine_similarity(eager_out, trt_out)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=(
            "TorchAO FP8 WOQ TRT outputs don't match eager. "
            f"Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}"
        ),
    )
    torch._dynamo.reset()


@pytest.mark.unit
@unittest.skipIf(importlib.util.find_spec("torchao") is None, "torchao not installed")
@unittest.skipIf(not _has_fp8_gpu(), "FP8 GPU (compute capability >= 9.0) is required")
def test_linear_fp8_static():
    import sys
    from pathlib import Path

    example_dir = (
        Path(__file__).resolve().parents[4] / "examples" / "dynamo" / "torchao"
    )
    sys.path.insert(0, str(example_dir))
    from static_fp8_utils import quantize_static_fp8

    class LinearModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = torch.nn.Linear(32, 64, bias=False)
            self.linear2 = torch.nn.Linear(64, 16, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear2(self.linear1(x))

    model = LinearModel().eval().to(dtype=torch.bfloat16, device="cuda")
    example_input = torch.randn(4, 32, dtype=torch.bfloat16, device="cuda")
    quantize_static_fp8(model, (example_input,), calibration_steps=3)

    exp_program = torch.export.export(model, (example_input,), strict=True)
    graph_str = str(exp_program.graph)
    assertions.assertIn(
        "quantize_affine_float8_non_decomposed",
        graph_str,
        "Exported graph is missing static FP8 quantize",
    )
    assertions.assertIn(
        "dequantize_affine_float8_non_decomposed",
        graph_str,
        "Exported graph is missing static FP8 dequantize",
    )

    trt_mod = torchtrt.dynamo.compile(
        exp_program,
        inputs=[example_input],
        enabled_precisions={torch.float8_e4m3fn},
        min_block_size=1,
        require_full_compilation=True,
        cache_built_engines=False,
        reuse_cached_engines=False,
    )
    with torch.no_grad():
        eager_out = model(example_input)
        trt_out = trt_mod(example_input)
    if isinstance(trt_out, (list, tuple)):
        trt_out = trt_out[0]
    cos_sim = cosine_similarity(eager_out, trt_out)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=(
            "TorchAO static FP8 TRT outputs don't match eager. "
            f"Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}"
        ),
    )
    torch._dynamo.reset()


@pytest.mark.unit
@unittest.skipIf(importlib.util.find_spec("torchao") is None, "torchao not installed")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
def test_linear_int4_woq():
    import sys
    from pathlib import Path

    example_dir = (
        Path(__file__).resolve().parents[4] / "examples" / "dynamo" / "torchao"
    )
    sys.path.insert(0, str(example_dir))
    from int4_utils import pre_process_model_for_export, quantize_linear_int4_symmetric
    from utils import exclude_dq_from_constant_folding

    class LinearModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(64, 128)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    model = LinearModel().eval().to(dtype=torch.bfloat16, device="cuda")
    example_input = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda")
    quantize_linear_int4_symmetric(model, group_size=32)
    model = pre_process_model_for_export(model)
    assertions.assertTrue(
        torch.all(model.linear.weight.zero_point == 0).item(),
        msg="INT4 WOQ test expected symmetric zero_point",
    )

    with exclude_dq_from_constant_folding():
        exp_program = torch.export.export(model, (example_input,), strict=True)

    dq_nodes = [
        n
        for n in exp_program.graph.nodes
        if n.target == torch.ops.torchao.dequantize_affine.default
    ]
    assertions.assertTrue(
        len(dq_nodes) >= 1,
        msg="Exported graph is missing torchao.dequantize_affine",
    )

    trt_mod = torchtrt.dynamo.compile(
        exp_program,
        inputs=[example_input],
        min_block_size=1,
        use_explicit_typing=True,
        require_full_compilation=True,
        immutable_weights=True,
        cache_built_engines=False,
        reuse_cached_engines=False,
    )

    with torch.no_grad():
        eager_out = model(example_input)
        trt_out = trt_mod(example_input)
    if isinstance(trt_out, (list, tuple)):
        trt_out = trt_out[0]
    cos_sim = cosine_similarity(eager_out, trt_out)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=(
            "TorchAO INT4 WOQ TRT outputs don't match eager. "
            f"Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}"
        ),
    )
    torch._dynamo.reset()


def _has_nvfp4_support() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import tensorrt as trt
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor  # noqa: F401

        return hasattr(trt.DataType, "FP4")
    except Exception:
        return False


@pytest.mark.unit
@unittest.skipIf(importlib.util.find_spec("torchao") is None, "torchao not installed")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
@unittest.skipIf(not _has_nvfp4_support(), "TensorRT FP4 DataType is required")
def test_linear_nvfp4_woq():
    import sys
    from pathlib import Path

    example_dir = (
        Path(__file__).resolve().parents[4] / "examples" / "dynamo" / "torchao"
    )
    sys.path.insert(0, str(example_dir))
    from nvfp4_utils import pre_process_model_for_export, quantize_linear_nvfp4

    class LinearModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(64, 128)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    model = LinearModel().eval().to(dtype=torch.bfloat16, device="cuda")
    example_input = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda")
    quantize_linear_nvfp4(model)
    model = pre_process_model_for_export(model)

    exp_program = torch.export.export(model, (example_input,), strict=True)

    dq_nodes = [
        n
        for n in exp_program.graph.nodes
        if n.target == torch.ops.torchao_trt.dequantize_nvfp4.default
    ]
    assertions.assertTrue(
        len(dq_nodes) >= 1,
        msg="Exported graph is missing torchao_trt.dequantize_nvfp4",
    )

    trt_mod = torchtrt.dynamo.compile(
        exp_program,
        inputs=[example_input],
        min_block_size=1,
        use_explicit_typing=True,
        require_full_compilation=True,
        immutable_weights=True,
        cache_built_engines=False,
        reuse_cached_engines=False,
    )

    with torch.no_grad():
        eager_out = model(example_input)
        trt_out = trt_mod(example_input)
    if isinstance(trt_out, (list, tuple)):
        trt_out = trt_out[0]
    cos_sim = cosine_similarity(eager_out, trt_out)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=(
            "TorchAO NVFP4 WOQ TRT outputs don't match eager. "
            f"Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}"
        ),
    )
    torch._dynamo.reset()


def _has_mxfp4_support() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import tensorrt as trt
        from torchao.prototype.mx_formats.mx_tensor import MXTensor  # noqa: F401

        return hasattr(trt.DataType, "FP4") and hasattr(trt.DataType, "E8M0")
    except Exception:
        return False


@pytest.mark.unit
@unittest.skipIf(importlib.util.find_spec("torchao") is None, "torchao not installed")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
@unittest.skipIf(
    not _has_mxfp4_support(), "TensorRT FP4 and E8M0 DataTypes are required"
)
def test_linear_mxfp4():
    import sys
    from pathlib import Path

    example_dir = (
        Path(__file__).resolve().parents[4] / "examples" / "dynamo" / "torchao"
    )
    sys.path.insert(0, str(example_dir))
    from mxfp4_utils import pre_process_model_for_export, quantize_linear_mxfp4

    class LinearModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(64, 128)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    model = LinearModel().eval().to(dtype=torch.bfloat16, device="cuda")
    example_input = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda")
    quantize_linear_mxfp4(model)
    model = pre_process_model_for_export(model)

    exp_program = torch.export.export(model, (example_input,), strict=True)

    dq_nodes = [
        n
        for n in exp_program.graph.nodes
        if n.target == torch.ops.torchao_trt.dequantize_mxfp4.default
    ]
    assertions.assertTrue(
        len(dq_nodes) >= 1,
        msg="Exported graph is missing torchao_trt.dequantize_mxfp4",
    )

    trt_mod = torchtrt.dynamo.compile(
        exp_program,
        inputs=[example_input],
        min_block_size=1,
        use_explicit_typing=True,
        require_full_compilation=True,
        immutable_weights=True,
        cache_built_engines=False,
        reuse_cached_engines=False,
    )

    with torch.no_grad():
        eager_out = model(example_input)
        trt_out = trt_mod(example_input)
    if isinstance(trt_out, (list, tuple)):
        trt_out = trt_out[0]
    cos_sim = cosine_similarity(eager_out, trt_out)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=(
            "TorchAO MXFP4 TRT outputs don't match eager. "
            f"Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}"
        ),
    )
    torch._dynamo.reset()
