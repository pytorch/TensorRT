"""
Bind CUDA streams to StreamGuard attributes on a stream-planned model.

Stream-planned modules carry one ``torch.classes.tensorrt.StreamGuard``
ScriptObject per stream slot, registered as a top-level attribute named
``_trt_stream_guard_<i>``.  At save time the guards' live cudaStream_t
handles are dropped (cudaStream_t is process-local).  On load the guards
are unbound until the user binds fresh streams — either explicitly via
this helper, or implicitly via ``StreamGuard(..., auto_bind=True)`` set at
plan time.

Two model shapes are supported:

  * ``torch.fx.GraphModule`` (ExportedProgram path) — guards are reachable
    via ``getattr(model, name)`` for each ``_trt_stream_guard_<i>`` attr.
  * ``torch._inductor.AOTIModelPackageLoader`` and similar AOTI handles —
    guards are reachable via ``loader.get_custom_objs()`` (added upstream
    in pytorch/pytorch#182149) which returns a name->ScriptObject map.

The returned list of streams must be kept alive by the caller for the
duration of model execution, otherwise the underlying cudaStream_t handles
may be destroyed under the bound guards.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch

_STREAM_GUARD_PREFIX = "_trt_stream_guard_"


def _iter_stream_guards(model: Any) -> Iterable[tuple[str, Any]]:
    """Yield (attr_name, StreamGuard) pairs for every guard on the model.

    Handles:
      - torch.fx.GraphModule / nn.Module — walks attributes
      - AOTI loader (object with get_custom_objs) — walks the custom-objs map
    """
    # AOTI loader path: pytorch/pytorch#182149 exposed get_custom_objs() on
    # AOTIModelPackageLoader which returns a Dict[str, ScriptObject] aliasing
    # the live torchbind constants embedded in the loaded .pt2.  Mutations
    # propagate to subsequent loader.run() calls.
    get_custom_objs = getattr(model, "get_custom_objs", None)
    if callable(get_custom_objs):
        for name, obj in get_custom_objs().items():
            if not isinstance(obj, torch._C.ScriptObject):
                continue
            try:
                if (
                    obj._type().qualified_name()
                    == "__torch__.torch.classes.tensorrt.StreamGuard"
                ):
                    yield name, obj
            except (AttributeError, RuntimeError):
                continue
        return

    # nn.Module path
    if isinstance(model, torch.nn.Module):
        for name in dir(model):
            if not name.startswith(_STREAM_GUARD_PREFIX):
                continue
            try:
                attr = getattr(model, name)
            except AttributeError:
                continue
            if isinstance(attr, torch._C.ScriptObject):
                yield name, attr
        return

    raise TypeError(
        f"bind_stream_plan_streams: unsupported model type {type(model)!r}. "
        "Expected nn.Module / GraphModule or an AOTI loader exposing "
        "get_custom_objs()."
    )


def bind_stream_plan_streams(
    model: Any,
    streams: list[torch.cuda.Stream] | dict[str, torch.cuda.Stream] | None = None,
) -> list[torch.cuda.Stream]:
    """Bind CUDA streams to every StreamGuard attribute on a stream-planned
    model.

    Args:
        model: A stream-planned ``torch.fx.GraphModule`` (or wrapping
            ``nn.Module``), or an AOTI-loaded model exposing
            ``get_custom_objs()``.
        streams:
            * ``None`` — materialize one fresh ``torch.cuda.Stream`` per
              guard, on the device recorded in each guard's metadata.
            * ``list[Stream]`` — bind in slot order; length must equal the
              number of guards.
            * ``dict[str, Stream]`` — bind by guard attribute name; missing
              names get fresh streams, unknown names raise ``ValueError``.

    Returns:
        List of streams bound, in slot order.  Caller must keep this list
        alive for as long as the model is in use, so the underlying
        cudaStream_t handles are not destroyed under the bound guards.

    Raises:
        ValueError: stream count mismatch (list form), unknown name
            (dict form), or device mismatch between stream and guard.
        RuntimeError: the model has no StreamGuard attributes (was the
            stream plan applied?).
    """
    guards = list(_iter_stream_guards(model))
    if not guards:
        raise RuntimeError(
            "bind_stream_plan_streams: no StreamGuard attributes found on the "
            "model. Did you apply_stream_plan() before saving?"
        )

    # Sort by slot index so positional binding is stable.  Names look like
    # _trt_stream_guard_0, _trt_stream_guard_1, ...; we sort by trailing int.
    def _slot_idx(name: str) -> int:
        try:
            return int(name.rsplit("_", 1)[-1])
        except ValueError:
            return -1

    guards.sort(key=lambda nv: _slot_idx(nv[0]))

    bound: list[torch.cuda.Stream] = []
    if streams is None:
        for _, guard in guards:
            s = torch.cuda.Stream(device=guard.device_index())
            # Skip explicit bind if the guard auto-bound on load.
            if not guard.is_bound():
                guard.bind(s.cuda_stream)
            bound.append(s)
    elif isinstance(streams, list):
        if len(streams) != len(guards):
            raise ValueError(
                f"bind_stream_plan_streams: got {len(streams)} streams but "
                f"model has {len(guards)} stream guards"
            )
        for (name, guard), s in zip(guards, streams):
            if s.device.index != guard.device_index():
                raise ValueError(
                    f"bind_stream_plan_streams: stream for {name!r} is on "
                    f"cuda:{s.device.index} but guard expects "
                    f"cuda:{guard.device_index()}"
                )
            guard.bind(s.cuda_stream)
            bound.append(s)
    elif isinstance(streams, dict):
        valid_names = {n for n, _ in guards}
        unknown = set(streams) - valid_names
        if unknown:
            raise ValueError(
                f"bind_stream_plan_streams: unknown guard names {sorted(unknown)!r}. "
                f"Valid names: {sorted(valid_names)}"
            )
        for name, guard in guards:
            if name in streams:
                s = streams[name]
                if s.device.index != guard.device_index():
                    raise ValueError(
                        f"bind_stream_plan_streams: stream for {name!r} is on "
                        f"cuda:{s.device.index} but guard expects "
                        f"cuda:{guard.device_index()}"
                    )
            else:
                s = torch.cuda.Stream(device=guard.device_index())
            guard.bind(s.cuda_stream)
            bound.append(s)
    else:
        raise TypeError(
            f"bind_stream_plan_streams: streams must be list, dict, or None; "
            f"got {type(streams)!r}"
        )

    return bound
