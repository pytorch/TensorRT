# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""The warning for TensorRT QDP AOT plugins that run outside CUDA graphs.

An engine with QDP AOT plugins that keeps running outside CUDA graphs, where
TensorRT reloads the plugins' modules on every run, logs a warning once. The
Python runtime's engine counts its own runs, also when a loaded exported
program calls it; with the C++ runtime, ``TorchTensorRTModule.forward`` counts
its eager calls.
"""

import importlib.util
import json
import logging
import sys
import types
import uuid
from types import SimpleNamespace

import pytest
import torch
import torch_tensorrt
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._subclasses.functional_tensor import FunctionalTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import FlopCounterMode
from torch_tensorrt.dynamo.runtime import _TorchTensorRTModule as runtime_module
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import TorchTensorRTModule
from torch_tensorrt.dynamo.runtime._TRTEngine import TRTEngine
from torch_tensorrt.runtime import _cudagraphs

_CALLS = runtime_module._PLUGIN_RELOAD_WARNING_CALLS
_HAS_TRITON = importlib.util.find_spec("triton") is not None
_CPP_RUNTIME = torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime

requires_gpu_plugins = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch_tensorrt.ENABLED_FEATURES.qdp_plugin
        and _HAS_TRITON
    ),
    reason="needs a CUDA GPU, TensorRT QDP plugins and Triton",
)
requires_python_runtime = pytest.mark.skipif(
    _CPP_RUNTIME,
    reason="the C++ runtime implements tensorrt::execute_engine out of Python's sight",
)


@pytest.fixture(autouse=True)
def cudagraphs_off():
    """Every test starts and ends with Torch-TensorRT's CUDA graph modes off."""
    torch_tensorrt.runtime.set_cudagraphs_mode(False)
    yield
    torch_tensorrt.runtime.set_cudagraphs_mode(False)


def _identity_meta(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


# ---------------------------------------------------------------------------
# Finding QDP AOT plugin layers (no GPU)
# ---------------------------------------------------------------------------

# Layer names of an engine with a plugin inside Myelin, as TensorRT 11.4's
# engine inspector lists them at the default verbosity: one named by
# Torch-TensorRT's plugin converter and one that nobody named.
_PLUGIN_LAYER = "[plugin_limits_test.transpose.default]-[/transpose]_myl0_2"
_UNNAMED_PLUGIN_LAYER = "(Unnamed Layer* 4) [PluginV3_V3QUICKAOT]_myl0_4"
_OTHER_LAYERS = [
    "__myl_Cast_myl0_0",
    "__myl_Mul_myl0_1",
    "[MATRIX_MULTIPLY]-[aten_ops.matmul.default]-[/matmul]_myl0_3",
    "Reshape output output0 to a scalar",
    "__myl_Add_myl0_5",
]


def _layer_info(*names, detailed=False):
    layers = [{"Name": name, "LayerType": "Myelin"} for name in names]
    return json.dumps(
        {"Layers": layers if detailed else list(names), "Bindings": ["x", "y"]}
    )


@pytest.mark.parametrize("detailed", [False, True])
def test_plugin_layers_are_found_by_name(detailed):
    info = _layer_info(
        _OTHER_LAYERS[0],
        _PLUGIN_LAYER,
        *_OTHER_LAYERS[1:],
        _UNNAMED_PLUGIN_LAYER,
        detailed=detailed,
    )
    assert runtime_module._qdp_aot_plugin_layers(info) == [
        _PLUGIN_LAYER,
        _UNNAMED_PLUGIN_LAYER,
    ]


@pytest.mark.parametrize(
    "info",
    [
        _layer_info(*_OTHER_LAYERS),
        _layer_info(
            "Reformatting CopyNode for Input Tensor 0 to "
            "(Unnamed Layer* 4) [PluginV3_V3QUICKAOT]",
            "Reformatting CopyNode for Output Tensor 0 to "
            "[plugin_limits_test.transpose.default]-[/transpose]",
        ),
        "",
        "not json",
        "[]",
        json.dumps({"Layers": 3}),
        json.dumps({"Layers": [None, 3, {"LayerType": "PluginV3"}]}),
    ],
    ids=[
        "no-plugins",
        "names-mentioning-plugins",
        "empty",
        "not-json",
        "list",
        "bad-layers",
        "bad-entries",
    ],
)
def test_engines_without_recognizable_plugin_layers_have_none(info):
    assert runtime_module._qdp_aot_plugin_layers(info) == []


def test_plugins_registered_only_for_jit_are_not_aot_layers(monkeypatch):
    registry = {
        "plugin_limits_test::jit": SimpleNamespace(aot_impl_func=None),
        "plugin_limits_test::aot": SimpleNamespace(aot_impl_func=lambda: None),
    }
    fake_lib = types.ModuleType("tensorrt.plugin._lib")
    fake_lib.QDP_REGISTRY = registry
    monkeypatch.setitem(sys.modules, "tensorrt.plugin._lib", fake_lib)
    jit, aot, unregistered = (
        f"[plugin_limits_test.{name}.default]-[/{name}]_myl0_{index}"
        for index, name in enumerate(["jit", "aot", "unregistered"])
    )
    # A JIT plugin can only run when registered, so an unregistered one is AOT.
    assert runtime_module._qdp_aot_plugin_layers(
        _layer_info(jit, aot, unregistered)
    ) == [aot, unregistered]


# ---------------------------------------------------------------------------
# Which calls of TorchTensorRTModule.forward run the engine (no GPU)
# ---------------------------------------------------------------------------


def _answers(case):
    """What ``_executes_engine`` answers each time ``case`` asks it.

    ``case(probe, answers)`` either passes ``probe``, which asks about its
    input, to a tracer, or appends answers itself.
    """
    answers = []

    def probe(x):
        answers.append(runtime_module._executes_engine([x]))
        return x + 1

    case(probe, answers)
    assert answers, "the case never asked"
    return answers


def _in_mode(make_mode):
    """Ask about a real tensor while a fresh ``make_mode()`` is active."""

    def case(probe, answers):
        real = torch.zeros(2)
        with make_mode():
            answers.append(runtime_module._executes_engine([real]))

    return case


def _with_fake_input(probe, answers):
    with FakeTensorMode() as mode:
        fake = mode.from_tensor(torch.zeros(2))
    answers.append(runtime_module._executes_engine([fake]))


class _PassThroughMode(TorchDispatchMode):
    """A dispatch mode that runs every operator, as profilers and debuggers do."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize(
    "case",
    [
        lambda probe, _: torch.compile(probe, backend="eager")(torch.zeros(2)),
        lambda probe, _: torch.jit.trace(probe, torch.zeros(2), check_trace=False),
        lambda probe, _: make_fx(probe)(torch.zeros(2)),
        lambda probe, _: make_fx(probe, tracing_mode="fake")(torch.zeros(2)),
        lambda probe, _: make_fx(probe, pre_dispatch=True)(torch.zeros(2)),
        lambda probe, _: torch.func.functionalize(probe)(torch.zeros(2)),
        lambda probe, _: torch.func.vmap(probe)(torch.zeros(2, 2)),
        _in_mode(FakeTensorMode),
        _in_mode(FunctionalTensorMode),
        _with_fake_input,
    ],
    ids=[
        "torch.compile",
        "jit.trace",
        "make_fx",
        "make_fx-fake",
        "make_fx-pre-dispatch",
        "functionalize",
        "vmap",
        "fake-mode",
        "functional-mode",
        "fake-input",
    ],
)
def test_tracing_calls_do_not_run_the_engine(case):
    assert not any(_answers(case))


@pytest.mark.parametrize(
    "case",
    [
        lambda probe, _: probe(torch.zeros(2)),
        lambda probe, _: probe(torch.nn.Parameter(torch.zeros(2), requires_grad=False)),
        _in_mode(lambda: FlopCounterMode(display=False)),
        _in_mode(_PassThroughMode),
    ],
    ids=["eager", "parameter", "flop-counter", "pass-through-mode"],
)
def test_eager_calls_run_the_engine(case):
    # A tensor subclass or a dispatch mode that runs operators still runs the
    # engine, so such calls count and the check can end.
    assert all(_answers(case))


# ---------------------------------------------------------------------------
# The module's check, as with the C++ runtime (no GPU)
# ---------------------------------------------------------------------------


class _CppEngine:
    """Stands in for a C++-runtime engine, which runs out of Python's sight."""

    def __init__(self, layer_info):
        self.layer_info = layer_info
        self.inspections = 0

    def get_engine_layer_info(self):
        self.inspections += 1
        if isinstance(self.layer_info, Exception):
            raise self.layer_info
        return self.layer_info


class _DeviceTensor(torch.Tensor):
    """A host tensor that passes for a GPU one, so ``forward`` runs on the CPU."""

    @property
    def is_cuda(self):
        return True


def _device_tensor():
    return torch.Tensor._make_subclass(_DeviceTensor, torch.zeros(2))


def _plugin_layer_info():
    return _layer_info(_PLUGIN_LAYER, *_OTHER_LAYERS)


def _module(engine=None):
    """A one-input engine module, by default around a C++-runtime engine."""
    module = TorchTensorRTModule(name="_run_on_acc_0", input_binding_names=["x"])
    module.engine = _CppEngine(_plugin_layer_info()) if engine is None else engine
    return module


@pytest.fixture
def cpp_execute_engine(monkeypatch):
    """Stand in for the C++ runtime's ``tensorrt::execute_engine``."""
    runs = []

    def execute_engine(inputs, engine):
        runs.append(engine)
        return [inputs[0]]

    monkeypatch.setattr(torch.ops.tensorrt, "execute_engine", execute_engine)
    return runs


def _call(module, calls, x=None):
    """Make ``calls`` calls of the module's real ``forward``."""
    x = _device_tensor() if x is None else x
    for _ in range(calls):
        module(x)


def _reload_warnings(caplog):
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == runtime_module.__name__
        and record.levelno == logging.WARNING
        and "outside a CUDA graph" in record.getMessage()
    ]


@pytest.fixture
def warnings_log(caplog):
    caplog.set_level(logging.WARNING, logger=runtime_module.__name__)
    return caplog


def test_module_warns_once_after_repeated_calls_outside_cuda_graphs(
    warnings_log, cpp_execute_engine
):
    module = _module()

    _call(module, _CALLS - 1)
    assert _reload_warnings(warnings_log) == []
    assert module._plugin_reload_check_pending
    assert module._plugin_reload_layers == [_PLUGIN_LAYER]

    _call(module, 1)
    (message,) = _reload_warnings(warnings_log)
    assert "'_run_on_acc_0'" in message
    assert f"has run {_CALLS} times outside a CUDA graph" in message
    assert f"1 QDP AOT plugin layer(s), such as '{_PLUGIN_LAYER}'" in message
    assert "set_cudagraphs_mode(True)" in message
    assert not module._plugin_reload_check_pending

    _call(module, 3 * _CALLS)
    assert len(_reload_warnings(warnings_log)) == 1
    # The engine is inspected once, on its first call outside a CUDA graph.
    assert module.engine.inspections == 1
    assert len(cpp_execute_engine) == 4 * _CALLS


def test_module_without_plugins_settles_on_its_first_call(
    warnings_log, cpp_execute_engine
):
    module = _module(_CppEngine(_layer_info(*_OTHER_LAYERS)))
    _call(module, 1)
    assert not module._plugin_reload_check_pending
    assert module._plugin_reload_layers == []
    _call(module, 2 * _CALLS)
    assert _reload_warnings(warnings_log) == []
    assert module.engine.inspections == 1


def test_module_in_cuda_graph_mode_never_warns(warnings_log, cpp_execute_engine):
    torch_tensorrt.runtime.set_cudagraphs_mode(True)
    module = _module()
    _call(module, 2 * _CALLS)
    assert not module._plugin_reload_check_pending
    assert module.engine.inspections == 0
    assert _reload_warnings(warnings_log) == []


def test_whole_graph_mode_alone_does_not_end_the_check(
    warnings_log, cpp_execute_engine, monkeypatch
):
    # enable_cudagraphs sets this global mode while it wraps some module; the
    # wrapper's capture is a stream capture. A module that runs outside the
    # wrapper meanwhile reloads its plugins on every call.
    monkeypatch.setattr(
        _cudagraphs,
        "_PY_RT_CUDAGRAPHS",
        _cudagraphs.CudaGraphsMode.WHOLE_GRAPH_CUDAGRAPHS,
    )
    module = _module()
    _call(module, _CALLS)
    assert len(_reload_warnings(warnings_log)) == 1


def test_warm_up_calls_before_a_capture_do_not_warn(
    warnings_log, cpp_execute_engine, monkeypatch
):
    module = _module()
    _call(module, 3)
    assert module._plugin_reload_calls == 3

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    _call(module, 1)
    assert not module._plugin_reload_check_pending

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    _call(module, 2 * _CALLS)
    assert _reload_warnings(warnings_log) == []


def test_tracing_leaves_the_check_pending(
    warnings_log, cpp_execute_engine, monkeypatch
):
    module = _module()
    x = _device_tensor()
    with FakeTensorMode():
        _call(module, 2 * _CALLS, x)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    _call(module, 2 * _CALLS, x)

    assert module._plugin_reload_check_pending
    assert module._plugin_reload_calls == 0
    assert module.engine.inspections == 0
    assert _reload_warnings(warnings_log) == []


class _WatchedModule(TorchTensorRTModule):
    """Records each read of the flag that ``forward`` tests."""

    flag_reads = 0

    @property
    def _plugin_reload_check_pending(self):
        type(self).flag_reads += 1
        return self.__dict__.get("_pending", True)

    @_plugin_reload_check_pending.setter
    def _plugin_reload_check_pending(self, value):
        self.__dict__["_pending"] = value


def test_forward_reads_no_flag_while_compiling(cpp_execute_engine, monkeypatch):
    # Dynamo guards on what a traced forward reads, and the flag changes after
    # the first eager calls, so reading it there would recompile later.
    module = _WatchedModule(name="_run_on_acc_0", input_binding_names=["x"])
    module.engine = _CppEngine(_plugin_layer_info())
    _WatchedModule.flag_reads = 0

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    _call(module, 3)
    assert _WatchedModule.flag_reads == 0

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    _call(module, 1)
    assert _WatchedModule.flag_reads == 1


def test_failed_layer_query_never_fails_a_call(warnings_log, cpp_execute_engine):
    module = _module(_CppEngine(RuntimeError("inspector unavailable")))
    _call(module, 2 * _CALLS)
    assert not module._plugin_reload_check_pending
    assert _reload_warnings(warnings_log) == []


def test_tensorrt_rtx_never_warns(warnings_log, cpp_execute_engine, monkeypatch):
    monkeypatch.setattr(
        runtime_module,
        "ENABLED_FEATURES",
        runtime_module.ENABLED_FEATURES._replace(tensorrt_rtx=True),
    )
    module = _module()
    _call(module, 2 * _CALLS)
    assert not module._plugin_reload_check_pending
    assert module.engine.inspections == 0
    assert _reload_warnings(warnings_log) == []


def test_modules_pickled_before_the_warning_are_checked():
    # A module unpickled from an earlier release has none of the state.
    module = TorchTensorRTModule.__new__(TorchTensorRTModule)
    torch.nn.Module.__init__(module)
    assert module._plugin_reload_check_pending
    assert module._plugin_reload_layers is None
    assert module._plugin_reload_calls == 0


def test_a_new_engine_is_checked_afresh(warnings_log, cpp_execute_engine):
    module = _module()
    _call(module, _CALLS)
    assert not module._plugin_reload_check_pending

    # Pickling drops the state: the loaded module runs a new engine.
    state = module.__getstate__()
    assert not any(key.startswith("_plugin_reload") for key in state)

    # So does loading a state dict, which replaces the engine.
    module.set_extra_state((module.name, None, [], []))
    assert module._plugin_reload_check_pending
    assert module._plugin_reload_layers is None
    assert module._plugin_reload_calls == 0


# ---------------------------------------------------------------------------
# The Python runtime's engine counts its own runs (no GPU)
# ---------------------------------------------------------------------------


def _python_engine(layer_info=None):
    """A Python-runtime ``TRTEngine`` that runs ``x + 1`` instead of TensorRT."""
    engine = TRTEngine.__new__(TRTEngine)
    inspector = _CppEngine(_plugin_layer_info() if layer_info is None else layer_info)
    vars(engine).update(
        name="_run_on_acc_0",
        cudagraph=None,
        _has_nccl_ops=False,
        requires_output_allocator=False,
        use_output_allocator_outputs=False,
        _rtx_native_cudagraphs=False,
        get_engine_layer_info=inspector.get_engine_layer_info,
        _execute_standard=lambda inputs: inputs[0] + 1,
        inspector=inspector,
    )
    return engine


def _run(engine, runs):
    for _ in range(runs):
        engine.execute([torch.zeros(2)])


def test_python_runtime_engine_warns_once_after_repeated_runs(warnings_log):
    engine = _python_engine()
    _run(engine, _CALLS - 1)
    assert _reload_warnings(warnings_log) == []
    assert engine._plugin_reload_layers == [_PLUGIN_LAYER]

    _run(engine, 1)
    (message,) = _reload_warnings(warnings_log)
    assert "'_run_on_acc_0'" in message and _PLUGIN_LAYER in message
    assert not engine._plugin_reload_check_pending

    _run(engine, 2 * _CALLS)
    assert len(_reload_warnings(warnings_log)) == 1
    assert engine.inspector.inspections == 1


@requires_python_runtime
def test_engine_called_without_a_module_is_checked(warnings_log):
    # An exported program loaded with torch_tensorrt.load calls the op on its
    # engine directly, without a TorchTensorRTModule.
    engine = _python_engine()
    x = torch.zeros(2)
    for _ in range(2 * _CALLS):
        (output,) = torch.ops.tensorrt.execute_engine([x], engine)
    torch.testing.assert_close(output, x + 1)
    (message,) = _reload_warnings(warnings_log)
    assert f"has run {_CALLS} times outside a CUDA graph" in message


def test_python_runtime_engine_in_cuda_graph_mode_never_warns(warnings_log):
    torch_tensorrt.runtime.set_cudagraphs_mode(True)
    engine = _python_engine()
    _run(engine, 2 * _CALLS)
    assert not engine._plugin_reload_check_pending
    assert engine.inspector.inspections == 0
    assert _reload_warnings(warnings_log) == []


def test_python_runtime_engine_without_plugins_settles_on_its_first_run(
    warnings_log,
):
    engine = _python_engine(_layer_info(*_OTHER_LAYERS))
    _run(engine, 1)
    assert not engine._plugin_reload_check_pending
    _run(engine, 2 * _CALLS)
    assert engine.inspector.inspections == 1
    assert _reload_warnings(warnings_log) == []


@requires_python_runtime
def test_module_leaves_the_check_to_an_engine_that_counts_its_runs(warnings_log):
    engine = _python_engine()
    module = _module(engine)

    _call(module, 1)
    # The module's own check ends at once; the engine has counted the call.
    assert not module._plugin_reload_check_pending
    assert module._plugin_reload_calls == 0
    assert engine._plugin_reload_calls == 1

    _call(module, 2 * _CALLS)
    # One warning for the engine, not a second one for the module.
    assert len(_reload_warnings(warnings_log)) == 1
    assert engine.inspector.inspections == 1


# ---------------------------------------------------------------------------
# On the GPU: a real Triton plugin inside a real engine
# ---------------------------------------------------------------------------

if _HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _transpose_kernel(x_ptr, y_ptr, N: tl.constexpr):
        """Transposes an N x N tile through 4 * N * N bytes of shared memory."""
        rows = tl.arange(0, N)
        offsets = rows[:, None] * N + rows[None, :]
        tl.store(y_ptr + offsets, tl.trans(tl.load(x_ptr + offsets)))


def _register_transpose(size):
    """Register a Triton transpose of ``size`` x ``size`` tiles under a fresh name."""
    import torch_tensorrt.kernels as ttk

    name = f"transpose_{uuid.uuid4().hex[:12]}"
    ttk.triton_op(
        f"plugin_limits_test::{name}",
        kernel=_transpose_kernel,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32"},
        constexprs={"N": size},
        grid=lambda inputs, outputs: (1,),
        meta_fn=_identity_meta,
        supports_dynamic_shapes=False,
    )
    return name


class _PluginModel(torch.nn.Module):
    def __init__(self, op_name):
        super().__init__()
        self.op_name = op_name

    def forward(self, x):
        return getattr(torch.ops.plugin_limits_test, self.op_name)(x * 2.0) + 1.0


class _PlainModel(torch.nn.Module):
    def forward(self, x):
        return x * 2.0 + 1.0


def _compile(model, x):
    compiled = torch_tensorrt.compile(
        model.eval().cuda(), ir="dynamo", inputs=[x], min_block_size=1
    )
    (engine,) = [
        module
        for module in compiled.modules()
        if isinstance(module, TorchTensorRTModule)
    ]
    return compiled, engine


def _counter(engine_module):
    """What counts the engine's runs: its Python-runtime engine, or the module."""
    engine = engine_module.engine
    if getattr(type(engine), "_tracks_plugin_reloads", False):
        return engine
    return engine_module


def _restart_check(engine_module):
    """Start the plugin reload check of a shared engine module afresh."""
    engine_module._reset_plugin_reload_check()
    engine = engine_module.engine
    if getattr(type(engine), "_tracks_plugin_reloads", False):
        for name in (
            "_plugin_reload_check_pending",
            "_plugin_reload_layers",
            "_plugin_reload_calls",
        ):
            vars(engine).pop(name, None)


@pytest.fixture(scope="module")
def plugin_engine():
    """A compiled model whose single engine runs a 16 KB Triton plugin."""
    op_name = _register_transpose(64)
    x = torch.randn(64, 64, device="cuda")
    compiled, engine = _compile(_PluginModel(op_name), x)
    return compiled, engine, x, (x * 2.0).T + 1.0, op_name


@requires_gpu_plugins
def test_plugin_engine_warns_once_on_the_gpu(plugin_engine, warnings_log):
    compiled, engine, x, expected, op_name = plugin_engine
    _restart_check(engine)

    for _ in range(_CALLS - 1):
        torch.testing.assert_close(compiled(x), expected)
    assert _reload_warnings(warnings_log) == []
    layers = _counter(engine)._plugin_reload_layers
    assert len(layers) == 1 and f"plugin_limits_test.{op_name}.default" in layers[0]

    for _ in range(_CALLS):
        compiled(x)
    # One warning, though both the module and its engine run the check.
    (message,) = _reload_warnings(warnings_log)
    assert f"plugin_limits_test.{op_name}.default" in message


@requires_gpu_plugins
def test_plugin_engine_in_cuda_graph_mode_does_not_warn_on_the_gpu(
    plugin_engine, warnings_log
):
    compiled, engine, x, expected, _ = plugin_engine
    _restart_check(engine)

    torch_tensorrt.runtime.set_cudagraphs_mode(True)
    for _ in range(_CALLS + 1):
        output = compiled(x)
    torch.testing.assert_close(output, expected)
    assert not _counter(engine)._plugin_reload_check_pending
    assert _reload_warnings(warnings_log) == []


@requires_gpu_plugins
def test_plugin_engine_captured_after_warm_up_does_not_warn_on_the_gpu(
    plugin_engine, warnings_log
):
    compiled, engine, x, expected, _ = plugin_engine
    _restart_check(engine)

    static_input = x.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            compiled(static_input)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = compiled(static_input)
    for _ in range(_CALLS + 1):
        graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(static_output, expected)
    assert _counter(engine)._plugin_reload_calls == 3
    assert not _counter(engine)._plugin_reload_check_pending
    assert _reload_warnings(warnings_log) == []


@requires_gpu_plugins
def test_engine_without_plugins_is_inspected_once_on_the_gpu(warnings_log):
    x = torch.randn(8, 8, device="cuda")
    compiled, engine = _compile(_PlainModel(), x)

    for _ in range(_CALLS + 1):
        torch.testing.assert_close(compiled(x), x * 2.0 + 1.0)
    assert not _counter(engine)._plugin_reload_check_pending
    assert _counter(engine)._plugin_reload_layers == []
    assert _reload_warnings(warnings_log) == []


def _save_and_load(compiled, x, path):
    torch_tensorrt.save(
        compiled, str(path), output_format="exported_program", arg_inputs=[x]
    )
    loaded = torch_tensorrt.load(str(path)).module()
    # The loaded program calls tensorrt::execute_engine on its engines itself.
    assert not any(isinstance(m, TorchTensorRTModule) for m in loaded.modules())
    return loaded


@requires_gpu_plugins
@requires_python_runtime
def test_loaded_exported_program_warns_once_on_the_gpu(
    plugin_engine, warnings_log, tmp_path
):
    compiled, _, x, expected, op_name = plugin_engine
    loaded = _save_and_load(compiled, x, tmp_path / "plugin.ep")
    warnings_log.clear()

    for _ in range(_CALLS - 1):
        torch.testing.assert_close(loaded(x), expected)
    assert _reload_warnings(warnings_log) == []
    for _ in range(_CALLS):
        loaded(x)
    (message,) = _reload_warnings(warnings_log)
    assert f"plugin_limits_test.{op_name}.default" in message


@requires_gpu_plugins
@requires_python_runtime
def test_loaded_exported_program_in_cuda_graph_mode_does_not_warn_on_the_gpu(
    plugin_engine, warnings_log, tmp_path
):
    compiled, _, x, expected, _ = plugin_engine
    loaded = _save_and_load(compiled, x, tmp_path / "plugin.ep")
    warnings_log.clear()

    torch_tensorrt.runtime.set_cudagraphs_mode(True)
    for _ in range(_CALLS + 1):
        output = loaded(x)
    torch.testing.assert_close(output, expected)
    assert _reload_warnings(warnings_log) == []


@requires_gpu_plugins
def test_torch_compile_does_not_recompile_when_the_check_ends_on_the_gpu(
    plugin_engine,
):
    import torch._dynamo

    compiled, engine, x, expected, _ = plugin_engine
    _restart_check(engine)
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    try:
        wrapped = torch.compile(compiled, backend=backend)
        for _ in range(3):
            torch.testing.assert_close(wrapped(x), expected)
        # Eager calls end the module's check ...
        for _ in range(_CALLS + 1):
            compiled(x)
        assert not engine._plugin_reload_check_pending
        # ... which the compiled module does not guard on.
        for _ in range(3):
            torch.testing.assert_close(wrapped(x), expected)
        assert len(graphs) == 1
    finally:
        torch._dynamo.reset()


@requires_gpu_plugins
def test_a_new_engine_is_checked_again_on_the_gpu(plugin_engine, warnings_log):
    compiled, engine, x, expected, _ = plugin_engine
    _restart_check(engine)
    for _ in range(_CALLS):
        compiled(x)
    assert len(_reload_warnings(warnings_log)) == 1

    # Set the engine up again from its plan, as deferred engine setup does.
    engine.serialized_engine = engine.serialized_engine
    engine.engine = None
    engine.setup_engine()
    assert engine._plugin_reload_check_pending

    for _ in range(_CALLS):
        torch.testing.assert_close(compiled(x), expected)
    assert len(_reload_warnings(warnings_log)) == 2
