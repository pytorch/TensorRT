"""Unit tests for the legacy dynamo exporter's submodule inlining
(torch_tensorrt.dynamo._exporter). These run on plain fx graphs and need neither a
GPU nor a TensorRT build."""

import operator

import pytest
import torch
from torch_tensorrt.dynamo._exporter import inline_torch_modules


@pytest.mark.unit
def test_inline_torch_modules_wires_inputs_by_position():
    """inline_torch_modules must wire a _run_on_gpu submodule's inputs from the
    call_module args by POSITION, not by matching placeholder names to graph nodes.

    Regression: the old name-matching path bound a submodule input to a same-named
    but unrelated graph node, rewiring a consumer to the wrong producer (and, for a
    submodule mixing graph-input and computed-intermediate inputs, leaking the
    latter as spurious graph placeholders). Here the submodule's first input
    placeholder is named "y", colliding with the parent's second input "y" even
    though the first *argument* is the parent's "x"; positional wiring must ignore
    the collision. Subtraction makes the input order observable.
    """
    # Submodule: out = first - second. First placeholder deliberately named "y".
    sub_graph = torch.fx.Graph()
    first = sub_graph.placeholder("y")
    second = sub_graph.placeholder("z")
    sub_graph.output(sub_graph.call_function(torch.sub, (first, second)))
    submodule = torch.fx.GraphModule(torch.nn.Module(), sub_graph)

    # Parent: inputs (x, y); call _run_on_gpu_0(x, y) -> expected x - y.
    parent_graph = torch.fx.Graph()
    x = parent_graph.placeholder("x")
    y = parent_graph.placeholder("y")
    root = torch.nn.Module()
    root.add_module("_run_on_gpu_0", submodule)
    call = parent_graph.call_module("_run_on_gpu_0", (x, y))
    parent_graph.output(call)
    parent = torch.fx.GraphModule(root, parent_graph)

    n_placeholders_before = sum(1 for n in parent.graph.nodes if n.op == "placeholder")

    inline_torch_modules(parent)
    parent.recompile()

    # No spurious placeholders leaked by the inlining.
    assert (
        sum(1 for n in parent.graph.nodes if n.op == "placeholder")
        == n_placeholders_before
    )
    # No call_module node survives (the submodule was inlined).
    assert not any(n.op == "call_module" for n in parent.graph.nodes)
    # Positional wiring: first input <- x, second input <- y, so out == x - y.
    out = parent(torch.tensor(5.0), torch.tensor(3.0))
    assert torch.allclose(out, torch.tensor(2.0))


@pytest.mark.unit
def test_inline_torch_modules_preserves_all_submodule_outputs():
    """A multi-output _run_on_gpu submodule must keep every output wired to its
    consumer after inlining. Regression: a mis-wired input orphaned one submodule
    output, which dead-code elimination then pruned, leaving a downstream consumer
    (or, in the hybrid case, a TensorRT engine) short an output at runtime.
    """
    # Submodule returns (a + b, a - b); both outputs are consumed downstream.
    sub_graph = torch.fx.Graph()
    a = sub_graph.placeholder("a")
    b = sub_graph.placeholder("b")
    add = sub_graph.call_function(torch.add, (a, b))
    sub = sub_graph.call_function(torch.sub, (a, b))
    sub_graph.output((add, sub))
    submodule = torch.fx.GraphModule(torch.nn.Module(), sub_graph)

    parent_graph = torch.fx.Graph()
    x = parent_graph.placeholder("x")
    y = parent_graph.placeholder("y")
    root = torch.nn.Module()
    root.add_module("_run_on_gpu_0", submodule)
    call = parent_graph.call_module("_run_on_gpu_0", (x, y))
    o0 = parent_graph.call_function(operator.getitem, (call, 0))
    o1 = parent_graph.call_function(operator.getitem, (call, 1))
    # Consume both outputs: (a+b) * (a-b).
    parent_graph.output(parent_graph.call_function(torch.mul, (o0, o1)))
    parent = torch.fx.GraphModule(root, parent_graph)

    inline_torch_modules(parent)
    parent.recompile()

    # (x+y)*(x-y) == x^2 - y^2 ; with x=5, y=3 -> 25 - 9 = 16.
    out = parent(torch.tensor(5.0), torch.tensor(3.0))
    assert torch.allclose(out, torch.tensor(16.0))


@pytest.mark.unit
def test_inline_torch_modules_computed_intermediate_inputs():
    """A _run_on_gpu submodule whose inputs are computed intermediates (not top-level
    graph placeholders, and not name-matching any graph node) must inline correctly.
    This is the case the old zero-duplicate path handled; positional wiring preserves
    it (and it is the shape that leaked spurious placeholders in the mixed case).
    """
    # Submodule: out = m + n. Names don't collide with anything in the parent.
    sub_graph = torch.fx.Graph()
    m = sub_graph.placeholder("m")
    n = sub_graph.placeholder("n")
    sub_graph.output(sub_graph.call_function(torch.add, (m, n)))
    submodule = torch.fx.GraphModule(torch.nn.Module(), sub_graph)

    # Parent: x -> c0 = x*2, c1 = x+1; call _run_on_gpu_0(c0, c1) -> c0 + c1.
    parent_graph = torch.fx.Graph()
    x = parent_graph.placeholder("x")
    c0 = parent_graph.call_function(torch.mul, (x, 2))
    c1 = parent_graph.call_function(torch.add, (x, 1))
    root = torch.nn.Module()
    root.add_module("_run_on_gpu_0", submodule)
    call = parent_graph.call_module("_run_on_gpu_0", (c0, c1))
    parent_graph.output(call)
    parent = torch.fx.GraphModule(root, parent_graph)

    inline_torch_modules(parent)
    parent.recompile()

    # No spurious placeholders leaked; the computed intermediates stay in-graph.
    assert sum(1 for node in parent.graph.nodes if node.op == "placeholder") == 1
    # out = (x*2) + (x+1); x=5 -> 10 + 6 = 16.
    out = parent(torch.tensor(5.0))
    assert torch.allclose(out, torch.tensor(16.0))
