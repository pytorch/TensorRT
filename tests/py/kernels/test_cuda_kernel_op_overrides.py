"""Tests for cuda_kernel_op's override path and shared internal helpers.

The override path is what replaces the old ``custom_cuda_kernel_op`` /
``custom_plugin`` / ``cuda_python`` entry points: pass ``meta_fn`` /
``eager_fn`` / ``aot_fn`` / ``schema`` as keyword arguments to
:func:`torch_tensorrt.kernels.cuda_kernel_op` and the matching
``KernelSpec`` fields become optional.
"""

import pytest
import torch

import torch_tensorrt
import torch_tensorrt.kernels as ttk
from torch_tensorrt.kernels._register import _infer_schema

from .conftest import (
    SIGMOID_SRC,
    make_eager_sigmoid,
    make_sigmoid_aot,
    register_once,
    skip_no_cuda,
    skip_no_nvrtc,
    skip_no_qdp,
)

# ---- No-GPU: schema inference (small defs — needs real __annotations__) ----


def test_schema_single_tensor():
    def meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    s = _infer_schema(meta)
    assert "Tensor x" in s and "-> Tensor" in s


def test_schema_two_tensors():
    def meta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(a)

    s = _infer_schema(meta)
    assert "Tensor a" in s and "Tensor b" in s


def test_schema_mixed_scalar():
    def meta(x: torch.Tensor, scale: float) -> torch.Tensor:
        return torch.empty_like(x)

    s = _infer_schema(meta)
    assert "Tensor x" in s and "float scale" in s


def test_schema_inference_rejects_unsupported_annotations():
    def _bad_input(x: list) -> torch.Tensor:
        return torch.empty(1)

    def _bad_output(x: torch.Tensor) -> int:
        return 1

    with pytest.raises(ValueError, match="unsupported input annotation"):
        _infer_schema(_bad_input)
    with pytest.raises(ValueError, match="outputs must"):
        _infer_schema(_bad_output)


# ---- No-GPU: override-path plumbing ----


@skip_no_qdp
def test_overrides_forward_to_registrar(monkeypatch):
    """Override kwargs land on the shared precompiled registrar."""
    from torch_tensorrt.kernels import _derive, _register

    captured = {}
    monkeypatch.setattr(
        _register,
        "register_precompiled_qdp_plugin",
        lambda *a, **k: captured.update(k),
    )
    # Skip the real NVRTC compile — we're testing wiring, not codegen.
    monkeypatch.setattr(
        _derive,
        "_compile_kernel",
        lambda spec: (b"// fake ptx", None, None),
    )

    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    ttk.cuda_kernel_op(
        "ttk_test::override_forward",
        ttk.KernelSpec(
            kernel_source="// s",
            kernel_name="k",
            inputs=[ttk.InputDecl("x")],
        ),
        meta_fn=_meta,
        eager_fn=lambda x: x,
        aot_fn=lambda *a: None,
        schema="(Tensor x) -> Tensor",
        supports_dynamic_shapes=True,
    )

    assert captured["op_name"] == "ttk_test::override_forward"
    assert captured["schema"] == "(Tensor x) -> Tensor"
    assert captured["supports_dynamic_shapes"] is True
    assert captured["ptx"] == b"// fake ptx"
    assert captured["kernel_name"] == "k"
    assert callable(captured["aot_fn"])
    # User-supplied aot_fn always takes the AOT path.
    assert captured["use_aot_if_available"] is True


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({}, "meta_fn is not provided"),
        ({"meta_fn": lambda x: x}, "eager_fn or aot_fn is not provided"),
    ],
)
@skip_no_qdp
def test_override_missing_required_dsl_field(kwargs, match):
    """Omitting both the DSL field and its override raises at validation time."""
    spec = ttk.KernelSpec(
        kernel_source="// s",
        kernel_name="k",
        inputs=[ttk.InputDecl("x")],
        # outputs / geometry deliberately absent
    )
    with pytest.raises(ValueError, match=match):
        ttk.cuda_kernel_op("ttk_test::missing", spec, **kwargs)


def test_precompiled_qdp_registrar_skips_nvrtc(monkeypatch):
    """The common precompiled path must never invoke NVRTC."""
    from torch_tensorrt.kernels import _nvrtc, _register

    def _fail(*a, **k):
        raise AssertionError("compile_to_ptx must NOT run on the precompiled path")

    monkeypatch.setattr(_nvrtc, "compile_to_ptx", _fail)
    monkeypatch.setattr(
        _register, "_assert_registration_name_available", lambda *a, **k: None
    )
    monkeypatch.setattr(_register, "_make_aot_impl", lambda *a, **k: lambda *args: None)
    for name in (
        "_register_pytorch_op",
        "custom_op",
    ):
        monkeypatch.setattr(_register, name, lambda *a, **k: None)

    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    _register.register_precompiled_qdp_plugin(
        op_name="ttk_test::ptx_reused",
        ptx=b".visible .entry k() { ret; }",
        kernel_name="k",
        aot_fn=lambda *a: None,
        eager_fn=lambda x: x,
        meta_fn=_meta,
    )


@pytest.mark.parametrize(
    ("ptx", "kernel_name", "match"),
    [
        (b"", "k", "non-empty bytes"),
        (b"\xff", "k", "valid UTF-8"),
        (b"// no entry", "k", "does not define"),
        (b".visible .entry other() {}", "k", "does not define"),
        (b".visible .entry k() {}", "k\nother", "valid PTX entry identifier"),
    ],
)
def test_precompiled_qdp_rejects_invalid_ptx_before_registration(
    monkeypatch, ptx, kernel_name, match
):
    """Artifact errors must not leave Torch or TensorRT global registry state."""
    from torch_tensorrt.kernels import _register

    mutations = []
    monkeypatch.setattr(
        _register,
        "_assert_registration_name_available",
        lambda *a, **k: mutations.append("registry lookup"),
    )
    monkeypatch.setattr(
        _register,
        "_register_pytorch_op",
        lambda *a, **k: mutations.append("torch registration"),
    )
    monkeypatch.setattr(
        _register, "custom_op", lambda *a, **k: mutations.append("QDP registration")
    )

    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    with pytest.raises(ValueError, match=match):
        _register.register_precompiled_qdp_plugin(
            op_name="ttk_test::invalid_precompiled_ptx",
            ptx=ptx,
            kernel_name=kernel_name,
            aot_fn=lambda *a: None,
            eager_fn=None,
            meta_fn=_meta,
        )

    assert mutations == []


def test_precompiled_aot_rejects_scalar_torch_attributes_before_registration(
    monkeypatch,
):
    from torch_tensorrt.kernels import _register

    mutations = []
    monkeypatch.setattr(
        _register,
        "_assert_registration_name_available",
        lambda *a, **k: mutations.append("registry lookup"),
    )

    def _meta(x, alpha):
        return torch.empty_like(x)

    with pytest.raises(ValueError, match="cannot safely forward Torch scalar"):
        _register.register_precompiled_qdp_plugin(
            op_name="ttk_test::unsafe_aot_scalar",
            ptx=b".visible .entry k() { ret; }",
            kernel_name="k",
            aot_fn=lambda *a: None,
            eager_fn=None,
            meta_fn=_meta,
            schema="(Tensor x, float alpha) -> Tensor",
        )

    assert mutations == []


@pytest.mark.parametrize(
    "schema",
    [
        "(Tensor x, Tensor x) -> Tensor",
        "(Tensor outputs) -> Tensor",
        "(Tensor lambda) -> Tensor",
    ],
)
def test_qdp_schema_rejects_names_unsafe_for_generated_callbacks(schema):
    from torch_tensorrt.kernels._register import analyze_op_schema

    with pytest.raises(ValueError, match="invalid argument names"):
        analyze_op_schema(lambda *args: args[0], schema)


def test_qdp_schema_must_match_meta_function_arity():
    from torch_tensorrt.kernels._register import analyze_op_schema

    def _meta(x):
        return torch.empty_like(x)

    with pytest.raises(ValueError, match="meta_fn must accept"):
        analyze_op_schema(_meta, "(Tensor x, Tensor y) -> Tensor")


@pytest.mark.parametrize(
    "schema",
    [
        "(Tensor x, *, Tensor y) -> Tensor",
        "(Tensor x, *, float alpha=1.0) -> Tensor",
    ],
)
def test_qdp_schema_rejects_keyword_only_inputs(schema):
    from torch_tensorrt.kernels._register import analyze_op_schema

    def _meta(*args, **kwargs):
        return torch.empty_like(args[0])

    with pytest.raises(ValueError, match="keyword-only inputs"):
        analyze_op_schema(_meta, schema)


@skip_no_qdp
def test_precompiled_aot_callback_follows_schema_declaration_order():
    import inspect

    from torch_tensorrt.kernels._register import _make_aot_impl, analyze_op_schema

    def _meta(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(left)

    info = analyze_op_schema(_meta)
    callback = _make_aot_impl(
        info,
        ".visible .entry k() { ret; }",
        "k",
        lambda inputs, outputs, tactic: None,
    )

    assert list(inspect.signature(callback).parameters) == [
        "left",
        "right",
        "outputs",
        "tactic",
    ]


def test_register_pytorch_op_partial_failure_is_atomic(monkeypatch):
    """A failure inside _register_pytorch_op must not leave the op half-registered.

    Without atomic teardown, ``lib.define`` from the first attempt would leave
    ``torch.ops.<ns>.<name>`` populated, ``_torch_op_already_registered`` would
    short-circuit on retry, and the user would silently keep a broken state
    with no CUDA / fake impl.
    """
    from torch_tensorrt.kernels import _register

    op_name = "ttk_test::partial_recovery"

    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    real_register_fake = torch.library.register_fake
    call_count = {"n": 0}

    def _flaky_register_fake(op):
        def _inner(fn):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated upstream failure")
            return real_register_fake(op)(fn)

        return _inner

    monkeypatch.setattr(torch.library, "register_fake", _flaky_register_fake)

    # First attempt: simulated failure must propagate.
    with pytest.raises(RuntimeError, match="simulated upstream failure"):
        _register._register_pytorch_op(op_name, _meta, eager_fn=None)

    # After failure, the op must look un-registered so a retry can recover.
    assert not _register._torch_op_already_registered(op_name)

    # Second attempt: the rigged register_fake lets this one through.
    _register._register_pytorch_op(op_name, _meta, eager_fn=None)
    assert _register._torch_op_already_registered(op_name)


def test_register_pytorch_op_rejects_existing_mismatched_schema():
    """An existing name must not silently retain an incompatible dispatcher ABI."""
    from torch_tensorrt.kernels import _register

    op_name = "ttk_test::schema_collision"

    def _meta_one(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    def _meta_two(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    _register._register_pytorch_op(op_name, _meta_one, eager_fn=None)
    assert _register._register_pytorch_op(op_name, _meta_one, eager_fn=None) is None

    with pytest.raises(ValueError, match="already registered with schema"):
        _register._register_pytorch_op(op_name, _meta_two, eager_fn=None)


def test_qdp_failure_cleans_up_torch_registration(monkeypatch):
    """A downstream QDP failure must leave the Torch name available for retry."""
    from torch_tensorrt.kernels import _register

    op_name = "ttk_test::qdp_failure_cleanup"

    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    monkeypatch.setattr(
        _register, "_assert_registration_name_available", lambda *a, **k: None
    )
    monkeypatch.setattr(_register, "_make_aot_impl", lambda *a, **k: lambda *args: None)

    def _fail(*args, **kwargs):
        raise RuntimeError("simulated QDP failure")

    monkeypatch.setattr(_register, "custom_op", _fail)

    with pytest.raises(RuntimeError, match="simulated QDP failure"):
        _register.register_precompiled_qdp_plugin(
            op_name=op_name,
            ptx=b".visible .entry k() { ret; }",
            kernel_name="k",
            aot_fn=lambda *a: None,
            eager_fn=None,
            meta_fn=_meta,
        )

    assert not _register._torch_op_already_registered(op_name)


@skip_no_qdp
def test_post_qdp_failure_rolls_back_all_state_and_can_retry(monkeypatch):
    """A failure after QDP wiring must not permanently occupy the plugin name."""
    import tensorrt as trt
    import tensorrt.plugin as trtp
    from tensorrt.plugin._lib import QDP_CREATORS, QDP_REGISTRY

    from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
        DYNAMO_ATEN_CONVERTERS,
    )
    from torch_tensorrt.kernels import _register

    op_name = "ttk_retry::post_qdp_failure"
    namespace, name = op_name.split("::")

    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    real_custom_op = _register.custom_op
    call_count = {"n": 0}
    converter_target = {"value": None}

    def _fail_after_qdp_registration(*args, **kwargs):
        real_custom_op(*args, **kwargs)
        call_count["n"] += 1
        if call_count["n"] == 1:
            converter_target["value"] = getattr(
                getattr(torch.ops, namespace), name
            ).default
            raise RuntimeError("simulated post-QDP failure")

    monkeypatch.setattr(_register, "custom_op", _fail_after_qdp_registration)

    registration_kwargs = dict(
        op_name=op_name,
        ptx=b".visible .entry k() { ret; }",
        kernel_name="k",
        aot_fn=lambda *args: None,
        eager_fn=None,
        meta_fn=_meta,
    )

    with pytest.raises(RuntimeError, match="simulated post-QDP failure"):
        _register.register_precompiled_qdp_plugin(**registration_kwargs)

    assert not _register._torch_op_already_registered(op_name)
    assert op_name not in QDP_REGISTRY
    assert op_name not in QDP_CREATORS
    assert not hasattr(getattr(trtp.op, namespace, object()), name)
    assert trt.get_plugin_registry().get_creator(name, "1", namespace) is None
    assert converter_target["value"] not in DYNAMO_ATEN_CONVERTERS

    # The identical registration now succeeds, proving no hidden registry slot
    # from the failed attempt still reserves the qualified name.
    _register.register_precompiled_qdp_plugin(**registration_kwargs)
    assert _register._torch_op_already_registered(op_name)
    assert op_name in QDP_REGISTRY


@skip_no_qdp
def test_concurrent_same_name_registration_preserves_successful_state(monkeypatch):
    """A losing concurrent registration must not roll back the winner's state."""
    import threading

    import tensorrt as trt
    import tensorrt.plugin as trtp
    from tensorrt.plugin._lib import QDP_CREATORS, QDP_REGISTRY

    from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
        DYNAMO_ATEN_CONVERTERS,
    )
    from torch_tensorrt.kernels import _register

    op_name = "ttk_concurrent::same_name"
    namespace, name = op_name.split("::")

    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    registration_kwargs = dict(
        op_name=op_name,
        ptx=b".visible .entry k() { ret; }",
        kernel_name="k",
        aot_fn=lambda *args: None,
        eager_fn=None,
        meta_fn=_meta,
    )

    # Hold the first caller inside the QDP mutation stage. The registrar's lock
    # must remain owned for this entire interval, and the second caller must see
    # the completed first registration only after it acquires that same lock.
    real_custom_op = _register.custom_op
    first_inside_qdp = threading.Event()
    allow_first_to_finish = threading.Event()

    def _blocking_custom_op(*args, **kwargs):
        first_inside_qdp.set()
        if not allow_first_to_finish.wait(timeout=10):
            raise RuntimeError("timed out waiting to finish QDP registration")
        return real_custom_op(*args, **kwargs)

    monkeypatch.setattr(_register, "custom_op", _blocking_custom_op)

    outcomes = {}
    second_calling = threading.Event()

    def _register_in_thread(label, *, announce=False):
        if announce:
            second_calling.set()
        try:
            _register.register_precompiled_qdp_plugin(**registration_kwargs)
        except Exception as exc:
            outcomes[label] = exc
        else:
            outcomes[label] = None

    first = threading.Thread(target=_register_in_thread, args=("first",))
    second = threading.Thread(
        target=_register_in_thread,
        args=("second",),
        kwargs={"announce": True},
    )
    first.start()
    assert first_inside_qdp.wait(timeout=10)

    # Non-blocking acquisition from this thread is a deterministic assertion
    # that the transaction still owns the lock while custom_op mutates QDP.
    lock_was_free = _register._REGISTRATION_LOCK.acquire(blocking=False)
    if lock_was_free:
        _register._REGISTRATION_LOCK.release()

    second.start()
    assert second_calling.wait(timeout=10)
    allow_first_to_finish.set()
    first.join(timeout=10)
    second.join(timeout=10)

    assert not first.is_alive()
    assert not second.is_alive()
    assert not lock_was_free
    assert outcomes["first"] is None
    assert isinstance(outcomes["second"], ValueError)
    assert "already registered" in str(outcomes["second"])

    converter_target = getattr(getattr(torch.ops, namespace), name).default
    assert _register._torch_op_already_registered(op_name)
    assert op_name in QDP_REGISTRY
    assert op_name in QDP_CREATORS
    assert hasattr(getattr(trtp.op, namespace), name)
    assert trt.get_plugin_registry().get_creator(name, "1", namespace) is not None
    assert converter_target in DYNAMO_ATEN_CONVERTERS


# ---- GPU: NVRTC compilation ----


@skip_no_cuda
@skip_no_qdp
@skip_no_nvrtc
class TestNVRTC:
    def test_compiles_to_ptx(self):
        from torch_tensorrt.kernels._nvrtc import compile_to_ptx

        ptx, _, _ = compile_to_ptx(
            SIGMOID_SRC, "ttk_test_sigmoid", ["/usr/local/cuda/include"]
        )
        assert isinstance(ptx, bytes) and b"ttk_test_sigmoid" in ptx

    def test_invalid_source_raises(self):
        from torch_tensorrt.kernels._nvrtc import compile_to_ptx

        with pytest.raises(Exception):
            compile_to_ptx(
                "this is not valid CUDA !!!###", "bad", ["/usr/local/cuda/include"]
            )

    def test_arch_override_respected(self):
        from torch_tensorrt.kernels._nvrtc import compile_to_ptx

        arch = f"sm_{torch.cuda.get_device_capability()[0]}0"
        ptx, _, _ = compile_to_ptx(
            SIGMOID_SRC,
            "ttk_test_sigmoid",
            ["/usr/local/cuda/include"],
            arch_override=arch,
        )
        assert isinstance(ptx, bytes)


# ---- GPU: integration — register via override path, exercise eager + TRT ----


def _register_sigmoid_via_overrides(op_name: str) -> None:
    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    ttk.cuda_kernel_op(
        op_name,
        ttk.KernelSpec(
            kernel_source=SIGMOID_SRC,
            kernel_name="ttk_test_sigmoid",
            inputs=[ttk.InputDecl("x")],
        ),
        meta_fn=_meta,
        eager_fn=make_eager_sigmoid(),
        aot_fn=make_sigmoid_aot(),
        supports_dynamic_shapes=True,
    )


@skip_no_cuda
@skip_no_qdp
@skip_no_nvrtc
class TestOverrideIntegration:
    def test_register_and_eager(self):
        register_once(
            "ttk_test::sigmoid_eager",
            lambda: _register_sigmoid_via_overrides("ttk_test::sigmoid_eager"),
        )
        x = torch.randn(1024, device="cuda")
        assert torch.allclose(
            torch.ops.ttk_test.sigmoid_eager(x), torch.sigmoid(x), atol=1e-4, rtol=1e-4
        )

    def test_trt_compile_dynamic_shapes(self):
        register_once(
            "ttk_test::sigmoid_dyn",
            lambda: _register_sigmoid_via_overrides("ttk_test::sigmoid_dyn"),
        )

        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.ttk_test.sigmoid_dyn(x)

        inputs = [
            torch_tensorrt.Input(
                min_shape=(1, 128),
                opt_shape=(1, 512),
                max_shape=(1, 2048),
                dtype=torch.float32,
            )
        ]
        trt = torch_tensorrt.compile(
            M().cuda().eval(),
            inputs=inputs,
            enabled_precisions={torch.float32},
            min_block_size=1,
        )
        for size in [128, 512, 2048]:
            x = torch.randn(1, size, device="cuda")
            with torch.no_grad():
                assert torch.allclose(trt(x), torch.sigmoid(x), atol=1e-2, rtol=1e-2)


@skip_no_cuda
@skip_no_qdp
@skip_no_nvrtc
def test_schema_override_integration():
    """End-to-end: schema= overrides the inferred schema at real registration."""
    src = """
    extern "C" __global__ void schema_ov_noop(
            const float* x, float alpha, float* y) {}
    """

    def _meta(x, alpha):  # no hints — only schema= makes ``float alpha`` land
        return torch.empty_like(x)

    ttk.cuda_kernel_op(
        "ttk_test::schema_ov",
        ttk.KernelSpec(
            kernel_source=src,
            kernel_name="schema_ov_noop",
            inputs=[ttk.InputDecl("x"), ttk.ScalarInput("alpha", float)],
            geometry=ttk.Elementwise(block=(256,), layout="flat"),
        ),
        meta_fn=_meta,
        eager_fn=lambda x, alpha: alpha * x,  # reference impl — kernel is a no-op
        schema="(Tensor x, float alpha) -> Tensor",
        supports_dynamic_shapes=True,
    )

    schemas = [
        str(s) for s in torch._C._jit_get_schemas_for_operator("ttk_test::schema_ov")
    ]
    assert any("float alpha" in s for s in schemas)

    x = torch.randn(32, device="cuda")
    assert torch.allclose(torch.ops.ttk_test.schema_ov(x, 2.5), 2.5 * x, atol=1e-5)
