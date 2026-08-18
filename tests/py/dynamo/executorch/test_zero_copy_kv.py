"""Export-side coverage for zero-copy aliased KV buffers.

Two halves have to agree for a zero-copy ``.pte`` to be correct:

  * ``rewire_aliased_mutations_to_buffers`` declares the buffer to be its own
    mutation result, which removes ExecuTorch's copy-back and takes the aliased
    output out of the delegate.
  * ``unstage_aliased_buffers_pass`` removes the host staging copy so the
    engine's in-place write lands in the caller's buffer.

The interesting failures are all silent -- a rewired mutation whose buffer is
still staged simply never updates -- so most of what is asserted here is which
mutations are left alone, which mis-shapes raise, and that a marked buffer that
is never un-staged is caught rather than dropped.
"""

import operator
from types import SimpleNamespace

import pytest

pytest.importorskip("executorch.exir")

import torch  # noqa: E402
import torch_tensorrt  # noqa: E402
from executorch.exir.backend.compile_spec_schema import CompileSpec  # noqa: E402
from executorch.exir.delegate import executorch_call_delegate  # noqa: E402
from executorch.exir.schema import DeviceType  # noqa: E402
from torch.export.exported_program import (  # noqa: E402
    OutputKind,
    OutputSpec,
    TensorArgument,
)
from torch_tensorrt.executorch import _zero_copy as Z  # noqa: E402
from torch_tensorrt.executorch.backend import (  # noqa: E402
    ZERO_COPY_KV_COMPILE_SPEC_KEY,
)

# The graphs below are built around torch.ops.tensorrt.execute_engine, which only
# exists once the Torch-TensorRT runtime operator library has loaded.
pytestmark = pytest.mark.skipif(
    not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime,
    reason="Torch-TensorRT runtime operators are not available",
)


def _patch_engine_metadata(monkeypatch, *, aliased_io, input_names, output_names):
    """Make every engine node report one fixed set of bindings and aliases."""
    import torch_tensorrt.dynamo.runtime._serialized_engine_layout as layout
    import torch_tensorrt.dynamo.runtime._TorchTensorRTModule as trt_module
    import torch_tensorrt.executorch._export_utils as export_utils

    info = ["x"] * (layout.ALIASED_IO_IDX + 1)
    info[layout.INPUT_BINDING_NAMES_IDX] = "IN"
    info[layout.OUTPUT_BINDING_NAMES_IDX] = "OUT"

    # The rewiring resolves engine info through _resolve_engine_info (the node is
    # still an execute_engine at this stage), so that is what to fake. The stub
    # requires metadata_only: without it the read goes through
    # TRTEngine.__getstate__ and re-serializes the whole engine to recover the
    # binding names and aliased_io, which are the only fields wanted here.
    def _fake_resolve(ep, node, *, metadata_only=False):
        assert metadata_only, "zero-copy reads binding metadata, not the engine"
        return info

    monkeypatch.setattr(export_utils, "_resolve_engine_info", _fake_resolve)
    monkeypatch.setattr(trt_module, "deserialize_aliased_io", lambda s: aliased_io)
    monkeypatch.setattr(
        layout,
        "deserialize_binding_names",
        lambda s: list(input_names) if s == "IN" else list(output_names),
    )


def _kv_program(*, mutation_value="aliased_getitem"):
    """A one-engine program: engine(k_buffer, tokens) -> (logits, k_out).

    ``mutation_value`` picks what the KV buffer's BUFFER_MUTATION is bound to:

      * ``"aliased_getitem"``: the engine's aliased output (what export
        declares for a caller-owned KV cache).
      * ``"user_getitem"``: a non-aliased engine output, the shape a copy-back
        mutation has.
      * ``"external_op"``: a value produced outside the engine.
    """
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    tokens = graph.placeholder("tokens")
    engine = graph.placeholder("engine")
    engine_call = graph.call_function(
        torch.ops.tensorrt.execute_engine.default, ([k_buffer, tokens], engine)
    )
    logits = graph.call_function(operator.getitem, (engine_call, 0))
    k_out = graph.call_function(operator.getitem, (engine_call, 1))
    mutation = {
        "aliased_getitem": k_out,
        "user_getitem": logits,
        "external_op": None,
    }[mutation_value]
    if mutation is None:
        mutation = graph.call_function(torch.add, (k_buffer, k_buffer))
    graph.output((mutation, logits))
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

    signature = SimpleNamespace(
        inputs_to_buffers={"b_k_0": "k_0"},
        input_specs=[],
        output_specs=[
            OutputSpec(
                OutputKind.BUFFER_MUTATION, TensorArgument(name=mutation.name), "k_0"
            ),
            OutputSpec(OutputKind.USER_OUTPUT, TensorArgument(name=logits.name), None),
        ],
    )
    program = SimpleNamespace(
        graph_module=graph_module,
        graph_signature=signature,
        _graph_signature=signature,
    )
    return program, k_buffer, k_out


@pytest.mark.unit
def test_rewire_points_the_mutation_at_its_buffer_and_marks_it(monkeypatch):
    """The aliased output is replaced by the buffer itself and then dies.

    With the mutation bound to the placeholder there is nothing for ExecuTorch
    to copy back, and with no other user the getitem leaves the graph -- which
    is what takes the aliased output out of the delegate. The elided output's
    binding name is returned so the backend can exempt exactly that one.
    """
    program, k_buffer, k_out = _kv_program()
    _patch_engine_metadata(
        monkeypatch,
        aliased_io={"out_k": ("k_in", "kv_cache_update")},
        input_names=["k_in", "tokens"],
        output_names=["logits", "out_k"],
    )

    assert Z.rewire_aliased_mutations_to_buffers(program) == ["out_k"]

    specs = program._graph_signature.output_specs
    assert specs[0].kind == OutputKind.BUFFER_MUTATION
    assert specs[0].target == "k_0"
    assert specs[0].arg.name == k_buffer.name
    output_node = program.graph_module.graph.output_node()
    assert output_node.args[0][0] is k_buffer
    assert k_out not in program.graph_module.graph.nodes
    assert k_buffer.meta["_torch_tensorrt_aliased_buffer"] is True


@pytest.mark.unit
@pytest.mark.parametrize("mutation_value", ["user_getitem", "external_op"])
def test_rewire_leaves_mutations_the_engine_does_not_alias(monkeypatch, mutation_value):
    """Only a mutation the engine satisfies in place may be rewired.

    A copy-back mutation ("user_getitem") and a mutation computed outside the
    engine ("external_op") both need their value copied into the buffer. Both
    look exactly like an aliased mutation in the graph, so the discriminator has
    to be the engine's own aliased_io -- rewiring either would delete a real
    update with no error.
    """
    program, k_buffer, _ = _kv_program(mutation_value=mutation_value)
    original_spec = program._graph_signature.output_specs[0]
    _patch_engine_metadata(
        monkeypatch,
        aliased_io={"out_k": ("k_in", "kv_cache_update")},
        input_names=["k_in", "tokens"],
        output_names=["logits", "out_k"],
    )

    assert Z.rewire_aliased_mutations_to_buffers(program) == []
    assert program._graph_signature.output_specs[0] is original_spec
    assert "_torch_tensorrt_aliased_buffer" not in k_buffer.meta


@pytest.mark.unit
def test_rewire_is_a_noop_without_aliased_io(monkeypatch):
    program, k_buffer, _ = _kv_program()
    _patch_engine_metadata(
        monkeypatch,
        aliased_io={},
        input_names=["k_in", "tokens"],
        output_names=["logits", "out_k"],
    )

    assert Z.rewire_aliased_mutations_to_buffers(program) == []
    assert "_torch_tensorrt_aliased_buffer" not in k_buffer.meta


@pytest.mark.unit
def test_rewire_rejects_an_engine_whose_every_output_is_aliased(monkeypatch):
    """A delegate with no outputs at all is not a shape anything supports.

    Nothing downstream reports it: the runtime reads elision off a single
    argument count, which a zero-output delegate satisfies, and the delegate
    itself is a pure node a later graph-wide dead-code elimination can erase.
    So the failure has to be raised here.
    """
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    engine = graph.placeholder("engine")
    engine_call = graph.call_function(
        torch.ops.tensorrt.execute_engine.default, ([k_buffer], engine)
    )
    k_out = graph.call_function(operator.getitem, (engine_call, 0))
    graph.output((k_out,))
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)
    signature = SimpleNamespace(
        inputs_to_buffers={"b_k_0": "k_0"},
        input_specs=[],
        output_specs=[
            OutputSpec(
                OutputKind.BUFFER_MUTATION, TensorArgument(name=k_out.name), "k_0"
            )
        ],
    )
    program = SimpleNamespace(
        graph_module=graph_module,
        graph_signature=signature,
        _graph_signature=signature,
    )
    _patch_engine_metadata(
        monkeypatch,
        aliased_io={"out_k": ("k_in", "kv_cache_update")},
        input_names=["k_in"],
        output_names=["out_k"],
    )

    with pytest.raises(RuntimeError, match="no outputs at all"):
        Z.rewire_aliased_mutations_to_buffers(program)


def _staged_delegate_graph(
    *, backend_id="TensorRTBackend", device=DeviceType.CUDA, compile_specs=None
):
    """A lowered graph: delegate(lowered, _h2d_copy(k_buffer), _h2d_copy(tokens))."""
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    tokens = graph.placeholder("tokens")
    lowered = graph.get_attr("lowered_module_0")
    h2d = torch.ops.et_copy._h2d_copy.default
    staged_k = graph.call_function(h2d, (k_buffer,))
    staged_tokens = graph.call_function(h2d, (tokens,))
    delegate = graph.call_function(
        executorch_call_delegate, (lowered, staged_k, staged_tokens)
    )
    graph.output((delegate,))

    root = torch.nn.Module()
    root.lowered_module_0 = SimpleNamespace(
        backend_id=backend_id, compile_specs=compile_specs
    )
    graph_module = torch.fx.GraphModule(root, graph)

    for node, spec_device in (
        (k_buffer, DeviceType.CPU),
        (tokens, DeviceType.CPU),
        (staged_k, device),
        (staged_tokens, device),
    ):
        node.meta["spec"] = SimpleNamespace(device=spec_device, device_index=3)
    return graph_module, k_buffer, staged_k, delegate


@pytest.mark.unit
def test_unstage_feeds_the_buffer_straight_to_the_delegate():
    """The marked buffer replaces its staging copy and moves to the device.

    Moving the spec is not cosmetic: memory planning reads it, and a buffer left
    in a host arena is somewhere the engine cannot write.
    """
    graph_module, k_buffer, staged_k, delegate = _staged_delegate_graph()
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    assert Z._unstage_aliased_buffers(graph_module) == 1

    assert delegate.args[1] is k_buffer
    assert k_buffer.meta["spec"].device == DeviceType.CUDA
    assert k_buffer.meta["spec"].device_index == 3
    # The other input is an ordinary one and keeps its staging copy.
    assert delegate.args[2] is not None
    assert delegate.args[2].target is torch.ops.et_copy._h2d_copy.default
    # The orphaned staging is erased, but only it -- the other staging survives.
    assert staged_k not in graph_module.graph.nodes


@pytest.mark.unit
def test_unstage_keeps_staging_for_an_unmarked_buffer():
    graph_module, k_buffer, staged_k, delegate = _staged_delegate_graph()

    assert Z._unstage_aliased_buffers(graph_module) == 0
    assert delegate.args[1] is staged_k
    assert k_buffer.meta["spec"].device == DeviceType.CPU


@pytest.mark.unit
def test_unstage_raises_for_a_marked_buffer_on_another_backends_delegate():
    """Only a TensorRT engine promises the in-place write, so a marked buffer
    routed to another backend's delegate is never un-staged -- and because
    export has already dropped its copy-back, that is a broken program, not a
    silent no-op."""
    graph_module, k_buffer, staged_k, delegate = _staged_delegate_graph(
        backend_id="CudaBackend"
    )
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    with pytest.raises(RuntimeError, match="no TensorRT delegate staging"):
        Z._unstage_aliased_buffers(graph_module)


@pytest.mark.unit
def test_unstage_raises_when_a_marked_buffer_is_never_unstaged():
    """A marked buffer that reaches no TensorRT delegate at all must raise, not
    return 0. Its copy-back is already gone, so leaving it staged would silently
    discard every update."""
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    graph.output((k_buffer,))
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)
    k_buffer.meta["spec"] = SimpleNamespace(device=DeviceType.CPU, device_index=0)
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    with pytest.raises(RuntimeError, match="no TensorRT delegate staging"):
        Z._unstage_aliased_buffers(graph_module)


@pytest.mark.unit
def test_unstage_raises_when_a_zero_copy_delegate_unstaged_nothing():
    """A TensorRT delegate that declares zero-copy but had no buffer un-staged
    (its mark did not survive to this pass) is unambiguously broken and must
    raise, naming the delegate."""
    graph_module, k_buffer, staged_k, delegate = _staged_delegate_graph(
        compile_specs=[CompileSpec(ZERO_COPY_KV_COMPILE_SPEC_KEY, b"[]")]
    )
    # k_buffer deliberately left unmarked: nothing gets un-staged for the delegate.

    with pytest.raises(RuntimeError, match="declares zero-copy KV"):
        Z._unstage_aliased_buffers(graph_module)


@pytest.mark.unit
def test_unstage_leaves_another_backends_same_gpu_staging_in_place():
    """A marked buffer read by a second backend on the *same* GPU is still moved.

    This is the accept side of the ``_h2d_copy`` allowance in
    ``_device_move_is_safe``: the other backend's delegate keeps its staging copy,
    which after the move reads a buffer already resident on the GPU it was copying
    to, so nothing it sees changes. The function's other accept branch, the graph
    ``output`` node, is pinned by
    ``test_unstage_allows_a_buffer_that_is_also_its_mutation_output``; every other
    unit test that gives the buffer a second ``_h2d_copy`` pins a refusal, so a
    rule that allowed no second staging at all would still pass all of those.
    """
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    h2d = torch.ops.et_copy._h2d_copy.default
    staged_trt = graph.call_function(h2d, (k_buffer,))
    staged_other = graph.call_function(h2d, (k_buffer,))
    lowered_trt = graph.get_attr("lowered_module_0")
    lowered_other = graph.get_attr("lowered_module_1")
    delegate_trt = graph.call_function(
        executorch_call_delegate, (lowered_trt, staged_trt)
    )
    delegate_other = graph.call_function(
        executorch_call_delegate, (lowered_other, staged_other)
    )
    graph.output((k_buffer, delegate_trt, delegate_other))
    root = torch.nn.Module()
    root.lowered_module_0 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=None
    )
    root.lowered_module_1 = SimpleNamespace(
        backend_id="CudaBackend", compile_specs=None
    )
    graph_module = torch.fx.GraphModule(root, graph)
    k_buffer.meta["spec"] = SimpleNamespace(device=DeviceType.CPU, device_index=0)
    staged_trt.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=0)
    staged_other.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=0)
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    assert Z._unstage_aliased_buffers(graph_module) == 1

    assert delegate_trt.args[1] is k_buffer
    assert delegate_other.args[1] is staged_other
    assert staged_other in graph_module.graph.nodes
    assert k_buffer.meta["spec"].device == DeviceType.CUDA
    assert k_buffer.meta["spec"].device_index == 0


@pytest.mark.unit
def test_unstage_raises_when_the_staging_copy_is_not_on_cuda():
    """Following the staging to the CPU would put the buffer out of the engine's
    reach, and the copy-back that would have saved it is already gone."""
    graph_module, k_buffer, _, _ = _staged_delegate_graph(device=DeviceType.CPU)
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    with pytest.raises(RuntimeError, match="not.*CUDA"):
        Z._unstage_aliased_buffers(graph_module)


@pytest.mark.unit
def test_unstage_raises_when_the_staging_copy_has_no_spec():
    graph_module, k_buffer, staged_k, _ = _staged_delegate_graph()
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True
    del staged_k.meta["spec"]

    with pytest.raises(RuntimeError, match="no TensorSpec"):
        Z._unstage_aliased_buffers(graph_module)


@pytest.mark.unit
def test_unstage_allows_a_buffer_that_is_also_its_mutation_output():
    """The zero-copy shape itself: the marked buffer is both the delegate's
    staged input and its own BUFFER_MUTATION graph output. The output-node
    reference carries no device of its own, so the device move must be allowed --
    the real lowered KV graph looks exactly like this."""
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    lowered = graph.get_attr("lowered_module_0")
    h2d = torch.ops.et_copy._h2d_copy.default
    staged_k = graph.call_function(h2d, (k_buffer,))
    delegate = graph.call_function(executorch_call_delegate, (lowered, staged_k))
    graph.output((k_buffer, delegate))
    root = torch.nn.Module()
    root.lowered_module_0 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=None
    )
    graph_module = torch.fx.GraphModule(root, graph)
    k_buffer.meta["spec"] = SimpleNamespace(device=DeviceType.CPU, device_index=3)
    staged_k.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=3)
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    assert Z._unstage_aliased_buffers(graph_module) == 1
    assert delegate.args[1] is k_buffer
    assert k_buffer.meta["spec"].device == DeviceType.CUDA


@pytest.mark.unit
def test_unstage_refuses_to_move_a_shared_buffer():
    """A buffer read by a consumer other than its TensorRT delegate staging
    cannot have its device moved -- that would silently retarget the other
    consumer too, exactly what ExecuTorch's PropagateDevicePass rejects.
    """
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    lowered = graph.get_attr("lowered_module_0")
    h2d = torch.ops.et_copy._h2d_copy.default
    staged_k = graph.call_function(h2d, (k_buffer,))
    other = graph.call_function(torch.add, (k_buffer, k_buffer))
    delegate = graph.call_function(executorch_call_delegate, (lowered, staged_k))
    graph.output((delegate, other))
    root = torch.nn.Module()
    root.lowered_module_0 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=None
    )
    graph_module = torch.fx.GraphModule(root, graph)
    k_buffer.meta["spec"] = SimpleNamespace(device=DeviceType.CPU, device_index=3)
    staged_k.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=3)
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    with pytest.raises(RuntimeError, match="consumer the move would disturb"):
        Z._unstage_aliased_buffers(graph_module)


@pytest.mark.unit
def test_unstage_refuses_to_move_a_buffer_staged_to_two_gpus():
    """One buffer staged to two TensorRT delegates on *different* GPUs cannot be
    un-staged for either: a spec carries one device index, so whichever engine
    lost the race would be handed an address on the other's GPU. ``spec.device``
    is only CUDA/CPU, so it is the device-index comparison in
    ``_device_move_is_safe`` that refuses the first delegate here. That is what
    separates this from the supported two-delegates-one-GPU shape, but it is not
    what this test pins: two branches of that function refuse the shape in
    sequence, and were the index comparison gone, delegate 0 would be un-staged
    and the direct-consumer branch would refuse delegate 1 instead, leaving this
    test green. The index comparison itself is pinned by
    ``test_unstage_refuses_a_buffer_another_backend_stages_to_a_different_gpu``.
    """
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    h2d = torch.ops.et_copy._h2d_copy.default
    staged_0 = graph.call_function(h2d, (k_buffer,))
    staged_1 = graph.call_function(h2d, (k_buffer,))
    lowered_0 = graph.get_attr("lowered_module_0")
    lowered_1 = graph.get_attr("lowered_module_1")
    delegate_0 = graph.call_function(executorch_call_delegate, (lowered_0, staged_0))
    delegate_1 = graph.call_function(executorch_call_delegate, (lowered_1, staged_1))
    graph.output((k_buffer, delegate_0, delegate_1))
    root = torch.nn.Module()
    root.lowered_module_0 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=None
    )
    root.lowered_module_1 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=None
    )
    graph_module = torch.fx.GraphModule(root, graph)
    k_buffer.meta["spec"] = SimpleNamespace(device=DeviceType.CPU, device_index=0)
    staged_0.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=0)
    staged_1.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=1)
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    with pytest.raises(RuntimeError, match="consumer the move would disturb"):
        Z._unstage_aliased_buffers(graph_module)


@pytest.mark.unit
def test_unstage_refuses_a_buffer_another_backend_stages_to_a_different_gpu():
    """A marked buffer staged to a TensorRT delegate on cuda:0 and to a
    *non*-TensorRT delegate on cuda:1 cannot be moved either.

    Un-staging skips the other backend's delegate, so its staging copy keeps
    reading the buffer while staging it to cuda:1, and re-homing the buffer onto
    the TensorRT engine's cuda:0 would move the source of that read to the wrong
    GPU. The device-index comparison is the only thing that refuses this shape:
    the other staging is an ``_h2d_copy`` like every supported one, and because
    it is never un-staged its delegate never becomes the kind of direct consumer
    the shared-buffer rule catches.
    """
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    h2d = torch.ops.et_copy._h2d_copy.default
    staged_trt = graph.call_function(h2d, (k_buffer,))
    staged_other = graph.call_function(h2d, (k_buffer,))
    lowered_trt = graph.get_attr("lowered_module_0")
    lowered_other = graph.get_attr("lowered_module_1")
    delegate_trt = graph.call_function(
        executorch_call_delegate, (lowered_trt, staged_trt)
    )
    delegate_other = graph.call_function(
        executorch_call_delegate, (lowered_other, staged_other)
    )
    graph.output((k_buffer, delegate_trt, delegate_other))
    root = torch.nn.Module()
    root.lowered_module_0 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=None
    )
    root.lowered_module_1 = SimpleNamespace(
        backend_id="CudaBackend", compile_specs=None
    )
    graph_module = torch.fx.GraphModule(root, graph)
    k_buffer.meta["spec"] = SimpleNamespace(device=DeviceType.CPU, device_index=0)
    staged_trt.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=0)
    staged_other.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=1)
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    with pytest.raises(RuntimeError, match="consumer the move would disturb"):
        Z._unstage_aliased_buffers(graph_module)


@pytest.mark.unit
def test_unstage_refuses_to_rehome_a_buffer_already_on_another_gpu():
    """A buffer already resident on cuda:0 is not re-homed to a second TensorRT
    delegate's cuda:1.

    Whether the move needs checking at all is decided by comparing the buffer's
    device *and index* against the staging copy's. Comparing the device alone
    would call this buffer already placed -- both ends are CUDA -- skip the
    check, and overwrite the index with the second engine's, leaving the first
    engine holding an address on the other GPU.
    """
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    h2d = torch.ops.et_copy._h2d_copy.default
    staged_0 = graph.call_function(h2d, (k_buffer,))
    staged_1 = graph.call_function(h2d, (k_buffer,))
    lowered_0 = graph.get_attr("lowered_module_0")
    lowered_1 = graph.get_attr("lowered_module_1")
    delegate_0 = graph.call_function(executorch_call_delegate, (lowered_0, staged_0))
    delegate_1 = graph.call_function(executorch_call_delegate, (lowered_1, staged_1))
    graph.output((k_buffer, delegate_0, delegate_1))
    root = torch.nn.Module()
    root.lowered_module_0 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=None
    )
    root.lowered_module_1 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=None
    )
    graph_module = torch.fx.GraphModule(root, graph)
    k_buffer.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=0)
    staged_0.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=0)
    staged_1.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=1)
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    with pytest.raises(RuntimeError, match="consumer the move would disturb"):
        Z._unstage_aliased_buffers(graph_module)
    assert k_buffer.meta["spec"].device_index == 0


@pytest.mark.unit
def test_zero_copy_backend_config_keeps_the_callers_config():
    """It composes onto a config rather than replacing one: a caller finalizing
    a zero-copy program still needs their own memory planning and passes."""
    from executorch.exir import ExecutorchBackendConfig

    inner = object()
    base = ExecutorchBackendConfig(to_out_var_pass=inner, emit_stacktrace=True)

    config = torch_tensorrt.executorch.zero_copy_backend_config(base)

    assert config.emit_stacktrace is True
    assert config.memory_planning_pass is base.memory_planning_pass
    assert config.to_out_var_pass is not inner
    # The caller's to_out_var_pass is not dropped, it is run after the un-staging.
    graph_module, k_buffer, _, delegate = _staged_delegate_graph()
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True
    seen = []
    base = ExecutorchBackendConfig(to_out_var_pass=lambda gm: seen.append(gm))
    torch_tensorrt.executorch.zero_copy_backend_config(base).to_out_var_pass.call(
        graph_module
    )
    assert seen == [graph_module]
    assert delegate.args[1] is k_buffer


@pytest.mark.unit
def test_zero_copy_backend_config_defaults_to_executorch_defaults():
    """Called with no config it starts from ExecuTorch's defaults, and the one
    field it replaces is to_out_var_pass, wrapped in the un-staging pass."""
    from executorch.exir import ExecutorchBackendConfig

    config = torch_tensorrt.executorch.zero_copy_backend_config()

    defaults = ExecutorchBackendConfig()
    # The un-staging pass specifically, not merely "some object that is not the
    # default" -- which is all any wrapper would have to be.
    assert type(config.to_out_var_pass).__name__ == "_UnstageThenToOutVar"
    assert type(config.memory_planning_pass) is type(defaults.memory_planning_pass)
    assert type(config.sym_shape_eval_pass) is type(defaults.sym_shape_eval_pass)
    assert config.emit_stacktrace == defaults.emit_stacktrace


@pytest.mark.unit
def test_unstage_pass_runs_the_inner_pass_after_unstaging():
    """A caller's own to_out_var_pass has to survive being composed with."""
    graph_module, k_buffer, _, delegate = _staged_delegate_graph()
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True
    seen = []

    def inner(gm):
        # The un-staging is already done by the time the inner pass sees the graph.
        seen.append(delegate.args[1] is k_buffer)
        return "inner-result"

    result = Z.unstage_aliased_buffers_pass(inner).call(graph_module)

    assert seen == [True]
    assert result == "inner-result"


# --------------------------------------------------------------------------
# Multi-delegate: a method that lowers to two TensorRT engines -- one with an
# aliased+elided KV buffer, one plain-compute engine with none. The zero-copy
# CompileSpec is appended once, to a single TensorRTPartitioner, and the
# partitioner must stamp it onto ONLY the delegate whose own engine had an
# aliased output elided. Stamped partition-wide instead, the plain delegate
# declares zero-copy while un-staging nothing, and the un-staging cross-check
# then rejects an otherwise-correct program.
# --------------------------------------------------------------------------


def _no_op_engine_node(graph, input_nodes, *, aliased_io, input_names, output_names):
    """A no_op_placeholder_for_execute_engine node with inlined engine info.

    Mirrors what replace_execute_engine() produces before partitioning: args are
    ``(input_list, *engine_info)`` with the binding names and aliased_io in their
    serialized wire form, so the partitioner's real per-engine resolution
    (_resolve_engine_info / _aliased_inputs_by_output_index) runs unmocked.
    """
    from torch_tensorrt.dynamo.runtime._serialized_engine_layout import (
        ALIASED_IO_IDX,
        DEVICE_IDX,
        ENGINE_IDX,
        INPUT_BINDING_NAMES_IDX,
        OUTPUT_BINDING_NAMES_IDX,
        SERIALIZATION_LEN,
        SERIALIZED_ENGINE_BINDING_DELIM,
    )
    from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import serialize_aliased_io

    info = [""] * SERIALIZATION_LEN
    info[ENGINE_IDX] = ""  # not read by the partitioner's elision resolution
    info[DEVICE_IDX] = "0"
    info[INPUT_BINDING_NAMES_IDX] = SERIALIZED_ENGINE_BINDING_DELIM.join(input_names)
    info[OUTPUT_BINDING_NAMES_IDX] = SERIALIZED_ENGINE_BINDING_DELIM.join(output_names)
    info[ALIASED_IO_IDX] = serialize_aliased_io(aliased_io)
    return graph.call_function(
        torch.ops.tensorrt.no_op_placeholder_for_execute_engine.default,
        (list(input_nodes), *info),
    )


def _two_engine_program():
    """engine_a(k_buffer, tokens) -> (logits, out_k[aliased]); engine_b(x) -> (y).

    ``k_buffer`` carries ``_torch_tensorrt_aliased_buffer`` (rewiring already
    ran); engine_a aliases its ``out_k`` output onto it, engine_b aliases nothing.
    """
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    tokens = graph.placeholder("tokens")
    x = graph.placeholder("x")
    engine_a = _no_op_engine_node(
        graph,
        [k_buffer, tokens],
        aliased_io={"out_k": ("k_in", "kv_cache_update")},
        input_names=["k_in", "tokens"],
        output_names=["logits", "out_k"],
    )
    engine_b = _no_op_engine_node(
        graph,
        [x],
        aliased_io={},
        input_names=["x_in"],
        output_names=["y"],
    )
    graph.output((engine_a, engine_b))
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)
    program = SimpleNamespace(
        graph_module=graph_module,
        graph_signature=SimpleNamespace(buffers_to_mutate={}, inputs_to_buffers={}),
        constants={},
    )
    return program, engine_a, engine_b


def _partition_two_engines(program, engine_a, engine_b, monkeypatch):
    """Run the real TensorRTPartitioner, one partition per engine node."""
    from torch_tensorrt.executorch.backend import _serialize_elided_output_names
    from torch_tensorrt.executorch.partitioner import TensorRTPartitioner

    class _FakeCap:
        def __init__(self, graph_module, *args, **kwargs):
            self._engines = [engine_a, engine_b]

        def propose_partitions(self):
            return [
                SimpleNamespace(id=i, nodes=[node])
                for i, node in enumerate(self._engines)
            ]

    monkeypatch.setattr(
        "torch_tensorrt.executorch.partitioner.CapabilityBasedPartitioner", _FakeCap
    )
    monkeypatch.setattr(
        "torch_tensorrt.executorch.partitioner.tag_constant_data",
        lambda exported_program: None,
    )
    # Appended once, method-wide -- exactly how export() builds the partitioner.
    partitioner = TensorRTPartitioner(
        compile_specs=[
            CompileSpec(
                ZERO_COPY_KV_COMPILE_SPEC_KEY,
                _serialize_elided_output_names(["out_k"]),
            )
        ]
    )
    return partitioner.partition(program)


def _zero_copy_names(compile_specs):
    from torch_tensorrt.executorch.backend import _elided_output_names

    return _elided_output_names(compile_specs)


@pytest.mark.unit
def test_partition_stamps_zero_copy_only_on_the_kv_delegate(monkeypatch):
    """The KV delegate carries the zero-copy spec naming its own elided binding,
    and the plain-compute delegate carries no zero-copy spec at all.

    The method-wide spec the partitioner is constructed with must not reach every
    partition: the names it holds are the method's, and only this engine's own
    aliased_io says which of them are its.
    """
    program, engine_a, engine_b = _two_engine_program()
    result = _partition_two_engines(program, engine_a, engine_b, monkeypatch)

    kv_specs = result.partition_tags["tensorrt_0"].compile_specs
    plain_specs = result.partition_tags["tensorrt_1"].compile_specs
    assert _zero_copy_names(kv_specs) == {"out_k"}
    assert _zero_copy_names(plain_specs) is None


@pytest.mark.unit
def test_multi_delegate_zero_copy_lowers_without_false_raise(monkeypatch):
    """A correct two-delegate zero-copy program survives the whole pipeline: run
    the real partitioner, build the lowered two-delegate graph from the specs it
    produced, and un-stage.

    The KV buffer is un-staged and the plain delegate is left alone. A plain
    delegate stamped zero-copy would instead make _unstage_aliased_buffers raise
    "declares zero-copy KV ... but no aliased buffer was un-staged for it" over a
    program that is correct.
    """
    program, engine_a, engine_b = _two_engine_program()
    result = _partition_two_engines(program, engine_a, engine_b, monkeypatch)
    kv_specs = result.partition_tags["tensorrt_0"].compile_specs
    plain_specs = result.partition_tags["tensorrt_1"].compile_specs

    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    x = graph.placeholder("x")
    h2d = torch.ops.et_copy._h2d_copy.default
    staged_k = graph.call_function(h2d, (k_buffer,))
    staged_x = graph.call_function(h2d, (x,))
    kv_lowered = graph.get_attr("lowered_module_0")
    plain_lowered = graph.get_attr("lowered_module_1")
    kv_delegate = graph.call_function(executorch_call_delegate, (kv_lowered, staged_k))
    plain_delegate = graph.call_function(
        executorch_call_delegate, (plain_lowered, staged_x)
    )
    graph.output((k_buffer, kv_delegate, plain_delegate))
    root = torch.nn.Module()
    root.lowered_module_0 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=kv_specs
    )
    root.lowered_module_1 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=plain_specs
    )
    graph_module = torch.fx.GraphModule(root, graph)
    for node, dev in (
        (k_buffer, DeviceType.CPU),
        (x, DeviceType.CPU),
        (staged_k, DeviceType.CUDA),
        (staged_x, DeviceType.CUDA),
    ):
        node.meta["spec"] = SimpleNamespace(device=dev, device_index=3)
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    assert Z._unstage_aliased_buffers(graph_module) == 1
    assert kv_delegate.args[1] is k_buffer
    # The plain delegate keeps its staging and is never demanded to un-stage.
    assert plain_delegate.args[1] is staged_x


@pytest.mark.unit
def test_unstage_raises_when_the_plain_delegate_is_wrongly_stamped():
    """The other side of the per-partition stamping, in isolation: a delegate
    that carries the zero-copy spec and un-stages nothing must raise, whatever
    put the spec there. Narrowing which delegates get stamped must not weaken
    this -- it is the lost-update guard for the KV delegate too.
    """
    zero_copy_spec = [CompileSpec(ZERO_COPY_KV_COMPILE_SPEC_KEY, b'["out_k"]')]
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    x = graph.placeholder("x")
    h2d = torch.ops.et_copy._h2d_copy.default
    staged_k = graph.call_function(h2d, (k_buffer,))
    staged_x = graph.call_function(h2d, (x,))
    kv_lowered = graph.get_attr("lowered_module_0")
    plain_lowered = graph.get_attr("lowered_module_1")
    kv_delegate = graph.call_function(executorch_call_delegate, (kv_lowered, staged_k))
    plain_delegate = graph.call_function(
        executorch_call_delegate, (plain_lowered, staged_x)
    )
    graph.output((k_buffer, kv_delegate, plain_delegate))
    root = torch.nn.Module()
    # Both delegates wrongly carry the spec -- the shape that per-engine stamping
    # in TensorRTPartitioner exists to prevent.
    root.lowered_module_0 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=list(zero_copy_spec)
    )
    root.lowered_module_1 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=list(zero_copy_spec)
    )
    graph_module = torch.fx.GraphModule(root, graph)
    for node, dev in (
        (k_buffer, DeviceType.CPU),
        (x, DeviceType.CPU),
        (staged_k, DeviceType.CUDA),
        (staged_x, DeviceType.CUDA),
    ):
        node.meta["spec"] = SimpleNamespace(device=dev, device_index=3)
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    with pytest.raises(RuntimeError, match="declares zero-copy KV"):
        Z._unstage_aliased_buffers(graph_module)


# --------------------------------------------------------------------------
# Single-engine, on the same engine-node helper: the same per-engine derivation
# also has to narrow *within* one engine, from every aliased output down to the
# ones whose aliased input is a buffer export rewired.
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_partition_elides_only_the_outputs_aliased_onto_a_marked_buffer():
    """One engine, two aliased outputs, one marked buffer: only the marked one is
    named elidable.

    An aliased output whose input is not a buffer export rewired -- a user alias,
    which nothing rewires and whose placeholder therefore carries no
    ``_torch_tensorrt_aliased_buffer`` -- is still a delegate output. Deriving the
    elidable set from the engine's aliased_io alone would exempt it too, and the
    backend would then accept a delegate that dropped a mutation nothing writes
    back.
    """
    from torch_tensorrt.executorch._zero_copy import _aliased_inputs_by_output_index
    from torch_tensorrt.executorch.partitioner import TensorRTPartitioner

    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    user_alias = graph.placeholder("u")
    engine = _no_op_engine_node(
        graph,
        [k_buffer, user_alias],
        aliased_io={
            "out_k": ("k_in", "kv_cache_update"),
            "out_u": ("u_in", "user"),
        },
        input_names=["k_in", "u_in"],
        output_names=["out_k", "out_u"],
    )
    graph.output((engine,))
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)
    program = SimpleNamespace(
        graph_module=graph_module,
        graph_signature=SimpleNamespace(buffers_to_mutate={}, inputs_to_buffers={}),
        constants={},
    )
    partition = SimpleNamespace(id=0, nodes=[engine])

    # Both outputs are aliased, or the narrowing below would have nothing to do.
    assert set(_aliased_inputs_by_output_index(program, engine)) == {0, 1}

    # The spec deliberately names the output the derivation must NOT pick, so a
    # result of {"out_k"} can only have come from the engine's aliased_io and the
    # marks on its inputs.
    partitioner = TensorRTPartitioner(
        compile_specs=[CompileSpec(ZERO_COPY_KV_COMPILE_SPEC_KEY, b'["out_u"]')]
    )
    assert partitioner._partition_elided_output_names(program, partition) == {"out_k"}


# --------------------------------------------------------------------------
# GPU integration: the mark set during rewiring must survive real lowering, or
# the un-staging pass has nothing to act on and every KV update is lost. Only a
# real export exercises that -- the stub graphs above set the mark by hand.
# --------------------------------------------------------------------------
VOCAB = 64
DIM = 32
HEADS = 2
HEAD_DIM = 16
MAX_LEN = 16


class _KVDecodeStep(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, DIM)
        self.pos_embed = torch.nn.Embedding(MAX_LEN, DIM)
        self.q = torch.nn.Linear(DIM, HEADS * HEAD_DIM, bias=False)
        self.k = torch.nn.Linear(DIM, HEADS * HEAD_DIM, bias=False)
        self.v = torch.nn.Linear(DIM, HEADS * HEAD_DIM, bias=False)
        self.o = torch.nn.Linear(HEADS * HEAD_DIM, DIM, bias=False)
        self.lm = torch.nn.Linear(DIM, VOCAB, bias=False)
        self.register_buffer("k_cache", torch.zeros(1, HEADS, MAX_LEN, HEAD_DIM))
        self.register_buffer("v_cache", torch.zeros(1, HEADS, MAX_LEN, HEAD_DIM))

    def forward(self, tokens: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        pos_idx = input_pos.reshape(-1)
        pos = input_pos.reshape(())
        x = self.embed(tokens) + self.pos_embed(input_pos.reshape(1, 1))

        def split_heads(proj: torch.Tensor) -> torch.Tensor:
            return proj.view(1, 1, HEADS, HEAD_DIM).transpose(1, 2)

        q = split_heads(self.q(x))
        k = split_heads(self.k(x))
        v = split_heads(self.v(x))
        self.k_cache.index_copy_(2, pos_idx, k)
        self.v_cache.index_copy_(2, pos_idx, v)
        scores = (q @ self.k_cache.transpose(-1, -2)) / (HEAD_DIM**0.5)
        allowed = torch.arange(MAX_LEN, device=x.device) <= pos
        bias = torch.where(
            allowed,
            torch.zeros((), dtype=x.dtype, device=x.device),
            torch.full((), torch.finfo(x.dtype).min, dtype=x.dtype, device=x.device),
        )
        attn = torch.softmax(scores + bias.view(1, 1, 1, MAX_LEN), dim=-1)
        out = (attn @ self.v_cache).transpose(1, 2).reshape(1, 1, HEADS * HEAD_DIM)
        return self.lm(self.o(out))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA + TensorRT for a real engine"
)
@pytest.mark.parametrize("generate_etrecord", [False, True], ids=["plain", "etrecord"])
def test_aliased_buffer_mark_survives_real_lowering(generate_etrecord):
    """After a real export(..., zero_copy_kv=True), the KV buffer placeholder in
    the lowered edge program still carries ``_torch_tensorrt_aliased_buffer`` --
    the token the to_out_var_pass keys the un-staging on.

    ``generate_etrecord=True`` is covered because it makes ExecuTorch deep copy
    the whole program, and the mark rides on node meta. Losing it there would not
    raise here: it surfaces later as the un-staging pass finding a marked buffer
    it never un-staged, by which point the connection to this option is gone.
    """
    with torch.no_grad():
        torch.manual_seed(0)
        model = _KVDecodeStep().eval().cuda()
        tokens = torch.zeros(1, 1, dtype=torch.long).cuda()
        input_pos = torch.tensor([0], dtype=torch.long).cuda()

        exported_program = torch.export.export(model, (tokens, input_pos))
        trt_gm = torch_tensorrt.dynamo.compile(
            exported_program,
            arg_inputs=(tokens, input_pos),
            min_block_size=1,
            truncate_double=True,
        )
        edge = torch_tensorrt.executorch.export(
            trt_gm,
            arg_inputs=(tokens, input_pos),
            retrace=False,
            zero_copy_kv=True,
            generate_etrecord=generate_etrecord,
        )

    ep = edge.exported_program()
    marked = [
        node
        for node in ep.graph_module.graph.nodes
        if node.op == "placeholder" and node.meta.get("_torch_tensorrt_aliased_buffer")
    ]
    assert marked, "no placeholder kept _torch_tensorrt_aliased_buffer through lowering"


# --------------------------------------------------------------------------
# save() path: unlike the direct export()+to_executorch() contract -- two paired
# calls the caller must not forget -- torch_tensorrt.save() owns both steps, so a
# single zero_copy_kv=True must both hand export() the opt-in and install the
# finalization config before to_executorch(). These are CPU-only: the TensorRT
# lowering and the ExecuTorch finalization are stubbed so the wiring is checked
# without a GPU. The passes they invoke have their own coverage above; a real
# end-to-end run is exercised by kv_cache_decode_check on GPU.
# --------------------------------------------------------------------------
def _trivial_exported_program():
    """A tiny CPU ExportedProgram -- enough for save() to reach _save_as_executorch.

    It carries no execute_engine node, so the retrace=True KV-declaration pass is
    a no-op on it and the stubs below stand in for the real lowering.
    """

    class _Add(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + 1

    return torch.export.export(_Add(), (torch.randn(3),))


def _install_save_stubs(monkeypatch, *, wrap_config=True):
    """Stub the executorch lowering that save() drives and record how it is called.

    Returns a namespace capturing the kwargs export() received, the arguments
    zero_copy_backend_config() was wrapped with, and the config finally handed to
    to_executorch(). When ``wrap_config`` is False the real
    zero_copy_backend_config runs, so the recorded config is the genuine one.
    """
    import torch_tensorrt._compile as compile_module
    import torch_tensorrt.executorch as executorch_api

    monkeypatch.setattr(
        compile_module,
        "ENABLED_FEATURES",
        compile_module.ENABLED_FEATURES._replace(torch_tensorrt_runtime=True),
    )

    calls = SimpleNamespace(
        export_kwargs=None, wrap_args=[], to_executorch_config="unset"
    )

    def _to_executorch(config=None):
        calls.to_executorch_config = config
        return SimpleNamespace(
            _tensor_data=None, write_to_file=lambda f: f.write(b"stub-pte")
        )

    edge = SimpleNamespace(to_executorch=_to_executorch)

    def _export(exp_program, **kwargs):
        calls.export_kwargs = kwargs
        return edge

    monkeypatch.setattr(executorch_api, "export", _export)

    if wrap_config:
        wrapped = object()

        def _wrap(config=None):
            calls.wrap_args.append(config)
            return wrapped

        monkeypatch.setattr(executorch_api, "zero_copy_backend_config", _wrap)
        calls.wrapped_sentinel = wrapped
    else:
        real_wrap = executorch_api.zero_copy_backend_config

        def _wrap(config=None):
            calls.wrap_args.append(config)
            return real_wrap(config)

        monkeypatch.setattr(executorch_api, "zero_copy_backend_config", _wrap)

    return calls


@pytest.mark.unit
def test_save_zero_copy_kv_true_threads_flag_and_installs_config(monkeypatch, tmp_path):
    """save(zero_copy_kv=True, backend_config=cfg) opts export() in and wraps the
    caller's config exactly once, forwarding the wrapped one to to_executorch()."""
    calls = _install_save_stubs(monkeypatch)
    user_cfg = object()

    torch_tensorrt.save(
        _trivial_exported_program(),
        str(tmp_path / "model.pte"),
        output_format="executorch",
        zero_copy_kv=True,
        backend_config=user_cfg,
    )

    assert calls.export_kwargs["zero_copy_kv"] is True
    # The user's config is wrapped once (preserving their fields), not double-wrapped.
    assert calls.wrap_args == [user_cfg]
    assert calls.to_executorch_config is calls.wrapped_sentinel


@pytest.mark.unit
def test_save_zero_copy_kv_true_wraps_defaults_without_a_config(monkeypatch, tmp_path):
    """With no backend_config, zero_copy_backend_config(None) starts from ET
    defaults; the finalization config is still installed."""
    calls = _install_save_stubs(monkeypatch)

    torch_tensorrt.save(
        _trivial_exported_program(),
        str(tmp_path / "model.pte"),
        output_format="executorch",
        zero_copy_kv=True,
    )

    assert calls.export_kwargs["zero_copy_kv"] is True
    assert calls.wrap_args == [None]
    assert calls.to_executorch_config is calls.wrapped_sentinel


@pytest.mark.unit
def test_save_zero_copy_kv_true_installs_the_real_unstaging_pass(monkeypatch, tmp_path):
    """End of the wiring with the real config builder: the config reaching
    to_executorch() carries the un-staging to_out_var_pass, not ET's default."""
    calls = _install_save_stubs(monkeypatch, wrap_config=False)

    torch_tensorrt.save(
        _trivial_exported_program(),
        str(tmp_path / "model.pte"),
        output_format="executorch",
        zero_copy_kv=True,
    )

    assert calls.wrap_args == [None]
    assert (
        type(calls.to_executorch_config.to_out_var_pass).__name__
        == "_UnstageThenToOutVar"
    )


@pytest.mark.unit
def test_save_defaults_leave_kv_staged(monkeypatch, tmp_path):
    """Default save() (zero_copy_kv omitted) never wraps the config, so the KV
    buffer keeps its staging and its copy-back: the caller's config reaches
    to_executorch() untouched and export() is told zero_copy_kv=False."""
    calls = _install_save_stubs(monkeypatch)
    user_cfg = object()

    torch_tensorrt.save(
        _trivial_exported_program(),
        str(tmp_path / "model.pte"),
        output_format="executorch",
        backend_config=user_cfg,
    )

    assert calls.export_kwargs["zero_copy_kv"] is False
    assert calls.wrap_args == []
    assert calls.to_executorch_config is user_cfg
