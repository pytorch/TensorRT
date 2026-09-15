"""Lifting a constant read by several get_attr nodes must produce one placeholder.

Lives here rather than with the exporter's other tests because this directory is
collected by the lane that runs on every pull request, and the tests for the rest
of the exporter are not.
"""

import pytest
import torch
from torch.export.graph_signature import (
    ExportGraphSignature,
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    TensorArgument,
)
from torch_tensorrt.dynamo._exporter import lift


@pytest.mark.unit
def test_lift_reuses_one_placeholder_per_constant():
    """A constant read by several nodes is lifted once.

    fx uniquifies a placeholder's name but not its target, and placeholder codegen
    emits the target, so lifting one constant twice yields a forward() with a
    duplicated argument that fails to compile with "SyntaxError: duplicate
    argument". One get_attr node with several users is fine; what breaks is several
    get_attr nodes carrying the same target, which is what the partitions left in
    PyTorch produce once the inliner copies them back into one graph.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    fake_mode = FakeTensorMode()
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    with fake_mode:
        x.meta["val"] = torch.empty(3, 4)
    # Two constants, each read by several get_attr nodes, which is what inlining
    # the partitions left in PyTorch produces. Two rather than one, so that reusing
    # whichever placeholder was lifted first is distinguishable from reusing the one
    # for this target.
    acc = x
    for target in ("a", "b", "a", "b"):
        acc = graph.call_function(torch.add, (acc, graph.get_attr(target)))
    last = acc
    graph.output((last,))

    root = torch.nn.Module()
    # Different values per constant, so reusing the placeholder for the wrong target
    # changes the result rather than producing the same answer by luck.
    root.register_buffer("a", torch.full((3, 4), 2.0))
    root.register_buffer("b", torch.full((3, 4), 5.0))
    gm = torch.fx.GraphModule(root, graph)

    graph_signature = ExportGraphSignature(
        input_specs=[
            InputSpec(InputKind.USER_INPUT, TensorArgument(name="x"), target=None)
        ],
        output_specs=[
            OutputSpec(
                OutputKind.USER_OUTPUT, TensorArgument(name=last.name), target=None
            )
        ],
    )

    lifted_gm, lifted_signature, _, _ = lift(gm, graph_signature)

    placeholders = [n for n in lifted_gm.graph.nodes if n.op == "placeholder"]
    targets = [str(n.target) for n in placeholders]
    assert len(targets) == len(set(targets)), f"duplicate placeholders: {targets}"

    # The user input plus one placeholder per constant, not one per read.
    assert len(placeholders) == 3, f"expected 3 placeholders, got {targets}"
    buffer_specs = [
        spec for spec in lifted_signature.input_specs if spec.kind == InputKind.BUFFER
    ]
    assert len(buffer_specs) == 2
    # Each spec must describe the placeholder in the same position, or the signature
    # and the arguments disagree even when the counts match.
    for spec, placeholder in zip(lifted_signature.input_specs, placeholders):
        assert (
            spec.arg.name == placeholder.name
        ), f"spec {spec.arg.name} does not match placeholder {placeholder.name}"

    # The duplicate argument surfaces when the graph is turned into python.
    lifted_gm.recompile()

    # Reusing the wrong placeholder passes every check above, so compare the value.
    # Reading a twice and b twice gives x + 2*(2 + 5).
    x_input = torch.ones(3, 4)
    a = root.get_buffer("a")
    b = root.get_buffer("b")
    torch.testing.assert_close(lifted_gm(a, b, x_input), (x_input + 2 * (a + b),))


@pytest.mark.unit
def test_inlined_partitions_produce_the_duplicate_reads_this_lifts():
    """The duplicate get_attr nodes the dedup handles come from real inlining.

    The test above builds them by hand, so on its own it would keep passing if the
    inliner stopped producing them and the dedup became dead code. This drives the
    partitioner and the inliner instead, so the shape under test stays tied to the
    thing that creates it. No engine is built and nothing is saved.
    """
    from torch_tensorrt.dynamo._exporter import inline_torch_modules
    from torch_tensorrt.dynamo.partitioning._adjacency_partitioner import partition

    class SharedBuffer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("w", torch.full((4, 4), 2.0))

        def forward(self, x):
            # Interleaved, so the buffer is read from several partitions that stay in
            # PyTorch. One partition reading it twice does not produce duplicates.
            a = torch.sub(x, self.w)
            b = torch.mul(a, a)
            c = torch.sub(b, self.w)
            d = torch.mul(c, c)
            return torch.sub(d, self.w)

    exported = torch.export.export(SharedBuffer().eval(), (torch.randn(4, 4),))
    partitioned, _ = partition(
        exported.module(),
        min_block_size=1,
        torch_executed_ops={"torch.ops.aten.sub.Tensor"},
    )
    inline_torch_modules(partitioned)

    targets = [str(n.target) for n in partitioned.graph.nodes if n.op == "get_attr"]
    assert len(targets) > len(
        set(targets)
    ), f"inlining no longer produces duplicate get_attr targets: {targets}"

    # Lifting this graph is left to the test above, which builds the same shape
    # directly. ep.module() also emits a guards node that lift cannot resolve, so
    # carrying this graph further would fail for a reason unrelated to the dedup.


def test_lift_renames_an_output_whose_node_fx_had_to_rename():
    """A lifted read that is also a graph output must keep its output spec pointing at
    the placeholder, including when fx renamed the get_attr node.

    The output spec holds the get_attr node's name. Keying the rename on the sanitised
    target instead only agrees when fx did not rename anything, so an attribute like
    ``W`` or ``myBuf`` left the spec naming a node that had just been erased and
    building the program raised SpecViolationError.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    for attr in ("W", "myBuf", "lower"):
        fake_mode = FakeTensorMode()
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        read = graph.get_attr(attr)
        with fake_mode:
            x.meta["val"] = torch.empty(3, 4)
            read.meta["val"] = torch.empty(3, 4)
        graph.output((read,))
        root = torch.nn.Module()
        setattr(root, attr, torch.ones(3, 4))
        gm = torch.fx.GraphModule(root, graph)

        signature = ExportGraphSignature(
            input_specs=[
                InputSpec(
                    kind=InputKind.USER_INPUT, arg=TensorArgument(name="x"), target=None
                )
            ],
            output_specs=[
                OutputSpec(
                    kind=OutputKind.USER_OUTPUT,
                    arg=TensorArgument(name=read.name),
                    target=None,
                )
            ],
        )
        lifted_gm, lifted_signature, _, _ = lift(gm, signature)

        placeholders = [n for n in lifted_gm.graph.nodes if n.op == "placeholder"]
        lifted_names = {n.name for n in placeholders}
        spec_name = lifted_signature.output_specs[0].arg.name
        assert spec_name in lifted_names, (
            f"attribute {attr!r}: the output spec names {spec_name!r}, which is not a "
            f"placeholder in the lifted graph {sorted(lifted_names)}. The spec still "
            "points at the erased get_attr node."
        )
