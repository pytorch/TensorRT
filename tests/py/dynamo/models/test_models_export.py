# type: ignore
import importlib
import platform
import unittest
from importlib import metadata

import pytest
import torch
import torch_tensorrt as torchtrt
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity

from packaging.version import Version

if importlib.util.find_spec("torchvision"):
    import torchvision.models as models

if importlib.util.find_spec("timm"):
    import timm

assertions = unittest.TestCase()


@pytest.mark.unit
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"), "torchvision not installed"
)
def test_resnet18(ir):
    model = models.resnet18(pretrained=True).eval().to("cuda")
    input = torch.randn((1, 3, 224, 224)).to("cuda")

    compile_spec = {
        "inputs": [
            torchtrt.Input(
                input.shape, dtype=torch.float, format=torch.contiguous_format
            )
        ],
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "ir": ir,
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "min_block_size": 8,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    trt_mod = torchtrt.compile(model, **compile_spec)
    cos_sim = cosine_similarity(model(input), trt_mod(input)[0])
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"Resnet18 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()


@unittest.skipIf(
    not importlib.util.find_spec("torchvision"), "torchvision not installed"
)
@pytest.mark.unit
def test_mobilenet_v2(ir):
    model = models.mobilenet_v2(pretrained=True).eval().to("cuda")
    input = torch.randn((1, 3, 224, 224)).to("cuda")

    compile_spec = {
        "inputs": [
            torchtrt.Input(
                input.shape, dtype=torch.float, format=torch.contiguous_format
            )
        ],
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "ir": ir,
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "min_block_size": 8,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    trt_mod = torchtrt.compile(model, **compile_spec)
    cos_sim = cosine_similarity(model(input), trt_mod(input)[0])
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"Mobilenet v2 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()


@pytest.mark.unit
@unittest.skipIf(
    not importlib.util.find_spec("timm") or not importlib.util.find_spec("torchvision"),
    "timm or torchvision not installed",
)
def test_efficientnet_b0(ir):
    model = timm.create_model("efficientnet_b0", pretrained=True).eval().to("cuda")
    input = torch.randn((1, 3, 224, 224)).to("cuda")

    compile_spec = {
        "inputs": [
            torchtrt.Input(
                input.shape, dtype=torch.float, format=torch.contiguous_format
            )
        ],
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "ir": ir,
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "min_block_size": 8,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    trt_mod = torchtrt.compile(model, **compile_spec)
    cos_sim = cosine_similarity(model(input), trt_mod(input)[0])
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"EfficientNet-B0 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()


@pytest.mark.unit
@unittest.skipIf(
    not importlib.util.find_spec("transformers"),
    "transformers is required to run this test",
)
def test_bert_base_uncased(ir):
    from transformers import BertModel

    model = (
        BertModel.from_pretrained("bert-base-uncased", return_dict=False).cuda().eval()
    )
    input = torch.randint(0, 2, (1, 14), dtype=torch.int32).to("cuda")
    input2 = torch.randint(0, 2, (1, 14), dtype=torch.int32).to("cuda")

    compile_spec = {
        "inputs": [
            torchtrt.Input(
                input.shape,
                dtype=input.dtype,
                format=torch.contiguous_format,
            ),
            torchtrt.Input(
                input.shape,
                dtype=input.dtype,
                format=torch.contiguous_format,
            ),
        ],
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "truncate_double": True,
        "ir": ir,
        "min_block_size": 10,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
        "attn_bias_is_causal": False,  # BERT uses bidirectional self-attention instead of causal
    }
    trt_mod = torchtrt.compile(model, **compile_spec)
    model_outputs = model(input, input2)
    trt_model_outputs = trt_mod(input, input2)
    assertions.assertTrue(
        len(model_outputs) == len(trt_model_outputs),
        msg=f"Number of outputs for BERT model compilation is different with Pytorch {len(model_outputs)} and TensorRT {len(trt_model_outputs)}. Please check the compilation.",
    )

    for index in range(len(model_outputs)):
        out, trt_out = model_outputs[index], trt_model_outputs[index]
        cos_sim = cosine_similarity(out, trt_out)
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"HF BERT base-uncased TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

    # Clean up model env
    torch._dynamo.reset()


@unittest.skipIf(
    not importlib.util.find_spec("torchvision"), "torchvision not installed"
)
@pytest.mark.unit
def test_resnet18_half(ir):
    model = models.resnet18(pretrained=True).eval().to("cuda").half()
    input = torch.randn((1, 3, 224, 224)).to("cuda").half()

    compile_spec = {
        "inputs": [
            torchtrt.Input(
                input.shape, dtype=torch.half, format=torch.contiguous_format
            )
        ],
        "device": torchtrt.Device("cuda:0"),
        "use_explicit_typing": False,
        "enabled_precisions": {torch.half},
        "ir": ir,
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "min_block_size": 8,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    trt_mod = torchtrt.compile(model, **compile_spec)
    cos_sim = cosine_similarity(model(input), trt_mod(input)[0])
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"Resnet18 Half TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()


@unittest.skipIf(
    torch.cuda.get_device_capability() < (10, 0),
    "FP4 quantization requires compute capability 10.0 or later",
)
@unittest.skipIf(
    not importlib.util.find_spec("modelopt"),
    "ModelOpt is required to run this test",
)
@unittest.skipIf(
    platform.system() != "Linux",
    "modelopt is only supported on Linux",
)
@pytest.mark.unit
def test_base_fp4_dynamic_shapes(ir):
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.utils import export_torch_mode

    dtype = torch.float16

    class SimpleNetwork(torch.nn.Module):
        def __init__(self):
            super(SimpleNetwork, self).__init__()
            self.linear1 = torch.nn.Linear(
                in_features=64, out_features=32, bias=True, dtype=dtype
            )

        def forward(self, x):
            x = self.linear1(x)
            return x

    def calibrate_loop(model):
        """Simple calibration function for testing."""
        model(dummy_inputs)

    BATCH_SIZE = torch.export.Dim("BATCH_SIZE", min=16, max=128)
    batch_size = 64
    dummy_inputs = torch.ones(batch_size, 64, dtype=dtype).cuda()

    model = SimpleNetwork().eval().cuda()

    quant_cfg = mtq.NVFP4_DEFAULT_CFG
    mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    # model has qdq nodes at this point
    with torch.no_grad():
        with export_torch_mode():
            exp_program = torch.export.export(
                model, (dummy_inputs,), strict=False, dynamic_shapes=({0: BATCH_SIZE},)
            )

            trt_model = torchtrt.dynamo.compile(
                exp_program,
                inputs=[dummy_inputs],
                min_block_size=1,
                cache_built_engines=False,
                reuse_cached_engines=False,
                use_explicit_typing=True,
            )
            batch_size = 128
            input_tensor = torch.ones(batch_size, 64, dtype=dtype).cuda()
            expected_output = model(input_tensor)
            outputs_trt = trt_model(input_tensor)
            abs_diff = torch.abs(expected_output - outputs_trt)
            print(f"max/mean abs_diff: {abs_diff.max().item()=} {abs_diff.mean()=}")
            assert torch.allclose(expected_output, outputs_trt, rtol=0.3, atol=0.3)


@unittest.skipIf(
    torch.cuda.get_device_capability() < (10, 0),
    "FP4 quantization requires compute capability 10.0 or later",
)
@unittest.skipIf(
    not importlib.util.find_spec("modelopt"),
    "ModelOpt is required to run this test",
)
@unittest.skipIf(
    platform.system() != "Linux",
    "modelopt is only supported on Linux",
)
@pytest.mark.unit
def test_base_fp4_static_shapes(ir):
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.utils import export_torch_mode

    dtype = torch.float16

    class SimpleNetwork(torch.nn.Module):
        def __init__(self):
            super(SimpleNetwork, self).__init__()
            self.linear1 = torch.nn.Linear(
                in_features=64, out_features=32, bias=True, dtype=dtype
            )

        def forward(self, x):
            x = self.linear1(x)
            return x

    def calibrate_loop(model):
        """Simple calibration function for testing."""
        model(input_tensor)

    input_tensor = torch.randn(128, 64, dtype=dtype).cuda()

    model = SimpleNetwork().eval().cuda()
    expected_output = model(input_tensor)

    quant_cfg = mtq.NVFP4_DEFAULT_CFG
    mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    # model has qdq nodes at this point
    with torch.no_grad():
        with export_torch_mode():
            exp_program = torch.export.export(model, (input_tensor,), strict=False)
            from torch.fx import passes

            trt_model = torchtrt.dynamo.compile(
                exp_program,
                inputs=[input_tensor],
                min_block_size=1,
                cache_built_engines=False,
                reuse_cached_engines=False,
                use_explicit_typing=True,
            )
            outputs_trt = trt_model(input_tensor)
            abs_diff = torch.abs(expected_output - outputs_trt)
            print(f"max/mean abs_diff: {abs_diff.max().item()=} {abs_diff.mean()=}")
            assert torch.allclose(expected_output, outputs_trt, rtol=0.3, atol=0.3)


@unittest.skipIf(
    torch.cuda.get_device_capability() < (8, 9),
    "FP8 quantization requires compute capability 8.9 or later",
)
@unittest.skipIf(
    not importlib.util.find_spec("modelopt"),
    "ModelOpt is required to run this test",
)
@unittest.skipIf(
    platform.system() != "Linux",
    "modelopt is only supported on Linux",
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
                min_block_size=1,
                cache_built_engines=False,
                reuse_cached_engines=False,
            )
            outputs_trt = trt_model(input_tensor)
            assert torch.allclose(output_pyt, outputs_trt, rtol=5e-3, atol=1e-2)


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@unittest.skipIf(
    not importlib.util.find_spec("modelopt"),
    "ModelOpt is required to run this test",
)
@unittest.skipIf(
    platform.system() != "Linux",
    "modelopt is only supported on Linux",
)
def test_base_int8(ir, dtype):
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

    input_tensor = torch.randn(1, 10).cuda().to(dtype)
    model = SimpleNetwork().eval().cuda().to(dtype)
    quant_cfg = mtq.INT8_DEFAULT_CFG
    mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    # model has INT8 qdq nodes at this point
    output_pyt = model(input_tensor)
    with torch.no_grad():
        with export_torch_mode():
            exp_program = torch.export.export(model, (input_tensor,), strict=False)
            with torchtrt.logging.debug():
                trt_model = torchtrt.dynamo.compile(
                    exp_program,
                    inputs=[input_tensor],
                    min_block_size=1,
                    cache_built_engines=False,
                    reuse_cached_engines=False,
                    truncate_double=True,
                    use_explicit_typing=True,
                )
            outputs_trt = trt_model(input_tensor)
            assert output_pyt.dtype == outputs_trt.dtype
            assert outputs_trt.dtype == dtype
            assert torch.allclose(output_pyt, outputs_trt, rtol=5e-3, atol=1e-2)


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@unittest.skipIf(
    not importlib.util.find_spec("modelopt"),
    "ModelOpt is required to run this test",
)
@unittest.skipIf(
    platform.system() != "Linux",
    "modelopt is only supported on Linux",
)
def test_base_int8_dynamic_shape(ir, dtype):
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.utils import export_torch_mode

    class SimpleNetwork(torch.nn.Module):
        def __init__(self):
            super(SimpleNetwork, self).__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)
            self.linear = torch.nn.Linear(222, 222)

        def forward(self, x):
            return self.linear(self.conv(x))

    def calibrate_loop(model):
        """Simple calibration function for testing."""
        model(input_tensor)

    BATCH_SIZE = torch.export.Dim("BATCH_SIZE", min=2, max=16)
    batch_size = 8
    input_tensor = torch.randn(batch_size, 3, 224, 224, dtype=dtype).cuda()
    model = SimpleNetwork().eval().cuda().to(dtype)

    quant_cfg = mtq.INT8_DEFAULT_CFG
    # RTX does not support INT8 default quantization(weights+activations), only support INT8 weights only quantization
    if torchtrt.tensorrt_package_name == "tensorrt_rtx":
        quant_cfg["quant_cfg"]["*input_quantizer"] = {"enable": False}
    mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)

    # model has INT8 qdq nodes at this point
    output_pyt = model(input_tensor)

    with torch.no_grad():
        with export_torch_mode():
            exp_program = torch.export.export(
                model, (input_tensor,), strict=False, dynamic_shapes=({0: BATCH_SIZE},)
            )
            trt_model = torchtrt.dynamo.compile(
                exp_program,
                inputs=[input_tensor],
                min_block_size=1,
                cache_built_engines=False,
                reuse_cached_engines=False,
                truncate_double=True,
                use_explicit_typing=True,
            )
            outputs_trt = trt_model(input_tensor)
            assert torch.allclose(output_pyt, outputs_trt, rtol=5e-2, atol=5e-2)


@unittest.skipIf(
    not importlib.util.find_spec("modelopt"),
    "ModelOpt is required to run this test",
)
@pytest.mark.unit
def test_fp8_mha_softmax_quantizer_annotation(ir):
    """Regression test for #4200: annotate_fp8_sdpa must tag an SDPA node whose
    Q, K, V inputs are all FP8-quantized via ``tensorrt.quantize_op``.

    This matches the FX pattern emitted by modelopt's
    ``register_attention_for_kv_quant`` when ``NVFP4_FP8_MHA_CONFIG`` is applied:
    the attention module's ``F.scaled_dot_product_attention`` call has its Q,
    K, V arguments wrapped by ``q_bmm_quantizer``, ``k_bmm_quantizer``,
    ``v_bmm_quantizer`` (all FP8).

    The annotated ``_fp8_softmax_scale = 1/448`` on the SDPA node lets the
    attention converter set ``IAttention.normalization_quantize_to_type = FP8``
    and ``IAttention.normalization_quantize_scale`` so TRT can fuse the full
    ``_gemm_mha_v2`` FP8 MHA kernel.

    Also verifies that INT8 Q/K/V (exponent_bits=0) or a partially-FP8 input
    (one of Q/K/V not quantized) do NOT trigger the annotation.
    """
    import torch.fx as fx
    from torch_tensorrt.dynamo._settings import CompilationSettings
    from torch_tensorrt.dynamo.lowering.passes.annotate_fp8_sdpa import (
        _SDPA_TARGETS,
        annotate_fp8_sdpa,
    )

    def _build_sdpa_input_quant_graph(
        exponent_bits: int, quantize_v: bool = True
    ) -> fx.GraphModule:
        """Build FX graph where Q, K, V flow into SDPA through quantize_op nodes."""
        graph = fx.Graph()
        q = graph.placeholder("q")
        k = graph.placeholder("k")
        v = graph.placeholder("v")
        amax = graph.placeholder("amax")
        q_q = graph.call_function(
            torch.ops.tensorrt.quantize_op.default,
            args=(q, amax, 8, exponent_bits, False, False),
        )
        k_q = graph.call_function(
            torch.ops.tensorrt.quantize_op.default,
            args=(k, amax, 8, exponent_bits, False, False),
        )
        v_q = (
            graph.call_function(
                torch.ops.tensorrt.quantize_op.default,
                args=(v, amax, 8, exponent_bits, False, False),
            )
            if quantize_v
            else v
        )
        out = graph.call_function(
            torch.ops.aten.scaled_dot_product_attention.default, args=(q_q, k_q, v_q)
        )
        graph.output(out)
        return fx.GraphModule({}, graph)

    settings = CompilationSettings()

    # FP8 Q/K/V inputs (exponent_bits=4): SDPA node must be annotated with 1/448.
    gm_fp8 = _build_sdpa_input_quant_graph(exponent_bits=4)
    annotate_fp8_sdpa(gm_fp8, settings)
    sdpa_nodes = [n for n in gm_fp8.graph.nodes if n.target in _SDPA_TARGETS]
    assert sdpa_nodes, "No SDPA node found in graph"
    assert all(
        "_fp8_softmax_scale" in n.meta for n in sdpa_nodes
    ), "annotate_fp8_sdpa did not annotate SDPA when Q/K/V inputs are FP8"
    expected_scale = 1.0 / 448.0
    for n in sdpa_nodes:
        assert (
            abs(n.meta["_fp8_softmax_scale"] - expected_scale) < 1e-12
        ), f"Wrong softmax scale: {n.meta['_fp8_softmax_scale']}"

    # INT8 Q/K/V inputs (exponent_bits=0): SDPA node must NOT be annotated.
    gm_int8 = _build_sdpa_input_quant_graph(exponent_bits=0)
    annotate_fp8_sdpa(gm_int8, settings)
    sdpa_int8 = [n for n in gm_int8.graph.nodes if n.target in _SDPA_TARGETS]
    assert all(
        "_fp8_softmax_scale" not in n.meta for n in sdpa_int8
    ), "annotate_fp8_sdpa incorrectly annotated SDPA when Q/K/V are INT8"

    # Only Q and K are FP8-quantized, V is raw: SDPA must NOT be annotated.
    gm_partial = _build_sdpa_input_quant_graph(exponent_bits=4, quantize_v=False)
    annotate_fp8_sdpa(gm_partial, settings)
    sdpa_partial = [n for n in gm_partial.graph.nodes if n.target in _SDPA_TARGETS]
    assert all(
        "_fp8_softmax_scale" not in n.meta for n in sdpa_partial
    ), "annotate_fp8_sdpa incorrectly annotated SDPA when V input is not FP8"


@unittest.skipIf(
    torch.cuda.get_device_capability() < (8, 9),
    "FP8 quantization requires compute capability 8.9 or later",
)
@pytest.mark.unit
def test_fp8_mha_fused_kernel(ir):
    """Regression test for #4200: FP8 MHA with FP8 Q/K/V inputs must produce a
    fused ``_gemm_mha_v2`` MHA kernel with normalization_quantize_to_type set.

    Hand-constructs the FX pattern that a future modelopt PyTorch-backend
    version will emit for FP8 MHA (mirrors PR NVIDIA/Model-Optimizer#1289):

        quantize_op(Q) ─┐
        quantize_op(K) ─┤─ scaled_dot_product_attention
        quantize_op(V) ─┘

    Built directly via ``torch.ops.tensorrt.quantize_op`` so we do not depend
    on modelopt actually supporting this pattern in its PyTorch backend today —
    if/when it does, torch-tensorrt will compile that graph to the fused kernel.

    Verifies:
    1. Engine inspector shows a layer name containing ``mha`` (i.e.
       ``_gemm_mha_v2``), confirming the FP8 MHA fusion triggered.
    2. Numerics match PyTorch reference SDPA within FP8 tolerance
       (cosine_similarity > 0.99).

    D=64 meets TRT's head_dim >= 32 requirement for the
    normalization_quantize FP8 kernel.
    """
    import json

    import torch_tensorrt

    import tensorrt as trt

    B, H, S, D = 1, 2, 32, 64
    torch.manual_seed(0)

    class FP8MHAModel(torch.nn.Module):
        """Mirror of what a modelopt FP8 MHA PyTorch export will look like:
        tensorrt.quantize_op on Q, K, V feeding F.scaled_dot_product_attention."""

        def __init__(self, amax_val: float = 6.0):
            super().__init__()
            self.register_buffer("amax_q", torch.tensor(amax_val, dtype=torch.float32))
            self.register_buffer("amax_k", torch.tensor(amax_val, dtype=torch.float32))
            self.register_buffer("amax_v", torch.tensor(amax_val, dtype=torch.float32))

        def forward(self, q, k, v):
            q_fp8 = torch.ops.tensorrt.quantize_op(q, self.amax_q, 8, 4, False, False)
            k_fp8 = torch.ops.tensorrt.quantize_op(k, self.amax_k, 8, 4, False, False)
            v_fp8 = torch.ops.tensorrt.quantize_op(v, self.amax_v, 8, 4, False, False)
            return torch.nn.functional.scaled_dot_product_attention(q_fp8, k_fp8, v_fp8)

    q = torch.randn(B, H, S, D, dtype=torch.float16).cuda()
    k = torch.randn(B, H, S, D, dtype=torch.float16).cuda()
    v = torch.randn(B, H, S, D, dtype=torch.float16).cuda()

    model = FP8MHAModel().eval().cuda()
    ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    exp_program = torch.export.export(model, (q, k, v), strict=False)
    serialized_engine = (
        torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
            exp_program,
            inputs=[q, k, v],
            use_explicit_typing=True,
            min_block_size=1,
        )
    )

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    inspector = engine.create_engine_inspector()
    engine_json = json.loads(
        inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    )
    layers = engine_json.get("Layers", [])
    layer_names = [
        layer if isinstance(layer, str) else layer.get("Name", "") for layer in layers
    ]
    assert any("mha" in name.lower() for name in layer_names), (
        f"No fused MHA kernel found in compiled engine. Expected a layer "
        f"containing 'mha' (e.g. _gemm_mha_v2) — TRT fuses FP8 Q/K/V + "
        f"normalization_quantize_to_type into a single MHA kernel. "
        f"Layer names present: {layer_names}"
    )

    # Numerical sanity: FP8-quantized MHA should agree with PyTorch SDPA.
    compiled = torch_tensorrt.compile(
        model,
        ir="dynamo",
        inputs=[q, k, v],
        use_explicit_typing=True,
        min_block_size=1,
    )
    with torch.no_grad():
        trt_out = compiled(q, k, v)
    cos = torch.nn.functional.cosine_similarity(
        ref_out.flatten().float().unsqueeze(0),
        trt_out.flatten().float().unsqueeze(0),
    ).item()
    assert (
        cos > 0.99
    ), f"FP8 MHA output deviates from PyTorch reference: cosine_similarity={cos}"
