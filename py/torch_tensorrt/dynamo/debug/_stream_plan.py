"""
Stream-plan visualizers, part of the Torch-TensorRT debugger.

  print_stream_plan(gm, plan)
      Prints a text summary of the stream assignment and cross-stream barriers.

  stream_plan_dot(gm, plan, *, title=...)  → pydot.Dot
      Returns a pydot.Dot graph object.

  show_stream_plan(gm, plan, *, path=None, fmt="png", title=...)
      Renders the graph to an image file inside the debug logging directory
      (torch_tensorrt_<user>/debug_logs/ by default).  Requires pydot.

Use Debugger.dump_stream_plan(gm, plan) to write directly into the active
debug session's logging_dir.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

import torch
import torch.fx
from torch_tensorrt.dynamo._defaults import DEBUG_LOGGING_DIR
from torch_tensorrt.runtime.stream_plan import StreamPlan, _trt_nodes

# ── Colour palette (8 hues, light enough for dark text) ──────────────────────

_PALETTE = [
    "#AED6F1",  # steel blue
    "#A9DFBF",  # sage green
    "#F9E79F",  # straw yellow
    "#F5CBA7",  # peach
    "#D2B4DE",  # lavender
    "#F1948A",  # salmon
    "#A3E4D7",  # aqua
    "#F8C471",  # amber
]
_CALLER_COLOR = "#F2F3F4"  # near-white for caller stream


# ── Helpers ───────────────────────────────────────────────────────────────────


def _stream_index(plan: StreamPlan) -> dict[int, int]:
    """Return {id(stream): 0-based-index} in assignment order (stable, deduped)."""
    seen: dict[int, int] = {}
    for stream in plan.assignment.values():
        if id(stream) not in seen:
            seen[id(stream)] = len(seen)
    return seen


def _node_label(n: torch.fx.Node) -> str:
    if n.op == "placeholder":
        return f"{n.name}\n[input]"
    if n.op == "output":
        return "output"
    if n.op == "call_module":
        return f"{n.target}\n[TRT]"
    return f"{n.name}\n[{n.op}]"


def _cross_stream_edges(
    gm: torch.fx.GraphModule,
    plan: StreamPlan,
) -> list[tuple[torch.fx.Node, torch.fx.Node, Optional[str], Optional[str]]]:
    """
    Return (pred, node, pred_stream_label, node_stream_label) for every edge
    that crosses a stream boundary (including caller → compute).
    """
    idx = _stream_index(plan)
    edges = []
    for n in gm.graph.nodes:
        n_stream = plan.assignment.get(n.target)
        n_label = None if n_stream is None else f"stream {idx[id(n_stream)]}"
        for pred in n.all_input_nodes:
            p_stream = plan.assignment.get(pred.target)
            if p_stream is n_stream:
                continue
            p_label = None if p_stream is None else f"stream {idx[id(p_stream)]}"
            edges.append((pred, n, p_label, n_label))
    return edges


# ── Text summary ─────────────────────────────────────────────────────────────


def print_stream_plan(
    gm: torch.fx.GraphModule,
    plan: StreamPlan,
    *,
    file: Optional[Any] = None,
) -> None:
    """Print a human-readable summary of the stream plan."""
    out = file or sys.stdout
    idx = _stream_index(plan)
    trt_list = _trt_nodes(gm)
    n_streams = len(idx)
    dev = plan.device or "cuda:?"

    def w(line: str = "") -> None:
        print(line, file=out)

    rule = "─" * 60
    w(
        f"Stream Plan  ──  {len(trt_list)} TRT subgraph(s), "
        f"{n_streams} unique stream(s)  (device: {dev})"
    )
    w()

    w("  Stream assignments")
    w(f"  {rule[:20]}")
    name_w = max((len(t) for t in plan.assignment), default=8)
    for target, stream in plan.assignment.items():
        si = idx[id(stream)]
        color_hint = f"  ({_PALETTE[si % len(_PALETTE)]})"
        w(
            f"    {target:<{name_w}}   stream {si}   handle 0x{stream.cuda_stream:x}{color_hint}"
        )
    w()

    cross = _cross_stream_edges(gm, plan)
    if cross:
        w("  Cross-stream barriers  (each requires a sync_streams event)")
        w(f"  {rule[:52]}")
        src_w = max((len(pl or "caller") for (_, _, pl, _) in cross), default=6)
        dst_w = max((len(nl or "caller") for (_, _, _, nl) in cross), default=6)
        for pred, n, pl, nl in cross:
            src = pl or "caller"
            dst = nl or "caller"
            via = f"  (edge: {pred.name} → {n.name})"
            w(f"    {src:<{src_w}}  →  {dst:<{dst_w}}{via}")
        w()
    else:
        w("  No cross-stream barriers needed (all subgraphs on same stream).")
        w()

    w("  Execution DAG  (★ = TRT engine,  × = sync barrier on this edge)")
    w(f"  {rule[:55]}")
    for n in gm.graph.nodes:
        if n.op in ("get_attr",):
            continue
        tag = "★ " if n.target in plan.assignment else "  "
        preds = [p.name for p in n.all_input_nodes]
        n_stream = plan.assignment.get(n.target)
        si_str = (
            f" [stream {idx[id(n_stream)]}]" if n_stream is not None else " [caller]"
        )
        if preds:
            edge_markers = []
            for pred in n.all_input_nodes:
                p_stream = plan.assignment.get(pred.target)
                marker = "×" if p_stream is not n_stream else "─"
                edge_markers.append(f"{pred.name}{marker}►")
            w(f"  {tag}{n.name}{si_str}")
            w(f"      {'  '.join(edge_markers)}")
        else:
            w(f"  {tag}{n.name}{si_str}")
    w()


# ── pydot graph ───────────────────────────────────────────────────────────────


def stream_plan_dot(
    gm: torch.fx.GraphModule,
    plan: StreamPlan,
    *,
    title: str = "Stream Plan",
) -> Any:
    """
    Return a pydot.Dot graph for the stream plan.

    Nodes are grouped into clusters by stream; data edges are solid black;
    cross-stream edges are dashed red with a sync label.

    Requires: pip install pydot
    """
    try:
        import pydot
    except ImportError as e:
        raise ImportError(
            "pydot is required for stream_plan_dot. "
            "Install it with: pip install pydot"
        ) from e

    idx = _stream_index(plan)
    trt_set = {n.target for n in _trt_nodes(gm)}
    n_streams = len(idx)
    dev = plan.device or "cuda:?"

    full_title = (
        f"{title} — {len(trt_set)} TRT subgraph(s), "
        f"{n_streams} stream(s), device: {dev}"
    )

    graph = pydot.Dot(
        graph_name="StreamPlan",
        graph_type="digraph",
        label=full_title,
        labelloc="t",
        fontname="Courier",
        fontsize=13,
        rankdir="TB",
        splines="spline",
        pad=0.4,
    )
    graph.set_node_defaults(
        fontname="Courier",
        fontsize=11,
        style="filled",
        shape="box",
        margin="0.2,0.1",
    )
    graph.set_edge_defaults(fontname="Courier", fontsize=10)

    # ── Caller-stream cluster ─────────────────────────────────────────────────
    caller_nodes = [
        n
        for n in gm.graph.nodes
        if n.op not in ("get_attr",) and n.target not in plan.assignment
    ]
    if caller_nodes:
        cluster = pydot.Cluster(
            "caller",
            label="caller stream",
            style="filled",
            fillcolor=_CALLER_COLOR,
            color="#AAAAAA",
            penwidth=1.5,
            fontname="Courier",
            fontsize=11,
        )
        for n in caller_nodes:
            cluster.add_node(
                pydot.Node(
                    n.name,
                    label=_node_label(n),
                    fillcolor=_CALLER_COLOR,
                )
            )
        graph.add_subgraph(cluster)

    # ── Per-stream clusters ───────────────────────────────────────────────────
    stream_targets: dict[int, list[str]] = {}
    for target, stream in plan.assignment.items():
        si = idx[id(stream)]
        stream_targets.setdefault(si, []).append(target)

    target_to_node = {n.target: n for n in gm.graph.nodes if n.op == "call_module"}

    unique_streams_ordered = sorted(
        {id(s): s for s in plan.assignment.values()}.items(),
        key=lambda kv: idx[kv[0]],
    )

    for stream_id_py, stream in unique_streams_ordered:
        si = idx[stream_id_py]
        color = _PALETTE[si % len(_PALETTE)]
        handle = f"0x{stream.cuda_stream:x}"
        cluster = pydot.Cluster(
            f"stream_{si}",
            label=f"stream {si}  |  handle {handle}",
            style="filled",
            fillcolor=color,
            color=color,
            penwidth=2,
            fontname="Courier",
            fontsize=11,
        )
        for target in stream_targets.get(si, []):
            n = target_to_node.get(target)
            if n is None:
                continue
            cluster.add_node(
                pydot.Node(
                    n.name,
                    label=_node_label(n),
                    fillcolor=color,
                )
            )
        graph.add_subgraph(cluster)

    # ── Edges ─────────────────────────────────────────────────────────────────
    for n in gm.graph.nodes:
        if n.op == "get_attr":
            continue
        n_stream = plan.assignment.get(n.target)
        for pred in n.all_input_nodes:
            if pred.op == "get_attr":
                continue
            p_stream = plan.assignment.get(pred.target)
            cross = p_stream is not n_stream
            if cross:
                if p_stream is None:
                    elbl = "sync\n(caller→compute)"
                elif n_stream is None:
                    elbl = "sync\n(compute→caller)"
                else:
                    si_src = idx[id(p_stream)]
                    si_dst = idx[id(n_stream)]
                    elbl = f"sync\n(s{si_src}→s{si_dst})"
                graph.add_edge(
                    pydot.Edge(
                        pred.name,
                        n.name,
                        style="dashed",
                        color="#CC3333",
                        fontcolor="#CC3333",
                        label=elbl,
                        penwidth=1.5,
                    )
                )
            else:
                graph.add_edge(pydot.Edge(pred.name, n.name))

    return graph


# ── Render to image ───────────────────────────────────────────────────────────


def show_stream_plan(
    gm: torch.fx.GraphModule,
    plan: StreamPlan,
    *,
    path: Optional[str] = None,
    fmt: str = "png",
    title: str = "Stream Plan",
) -> str:
    """
    Render the stream plan as an image.

    If `path` is given, save to that file.  Otherwise write to
    ``<DEBUG_LOGGING_DIR>/stream_plan.<fmt>``.

    Returns the path to the written file.

    Requires: pip install pydot
    """
    dot = stream_plan_dot(gm, plan, title=title)

    if path is None:
        os.makedirs(DEBUG_LOGGING_DIR, exist_ok=True)
        path = os.path.join(DEBUG_LOGGING_DIR, f"stream_plan.{fmt}")

    getattr(dot, f"write_{fmt}")(path)
    return path
