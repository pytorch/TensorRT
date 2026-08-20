"""Disk cache for Torch-TensorRT's lowered FX GraphModules.

The engine cache is later in the pipeline: its key is a TensorRT subgraph after
decomposition, lowering, and partitioning.  This cache is keyed from the
ExportedProgram before those passes so a warm compile can skip host-side graph
work and resume from the post-lowering ATen GraphModule.

The artifact is a torch.save of a node-list GraphModule (ops, args, names,
state_dict), not an ExportedProgram and not FX __reduce__ retrace. Retrace
rebuilds a different graph (node count/names) that breaks TRT conversion.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import operator
import os
import pickle
import tempfile
import time
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Sequence, Tuple

import tensorrt as trt
import torch
from torch_tensorrt._Input import Input
from torch_tensorrt._version import __version__ as torch_tensorrt_version
from torch_tensorrt.dynamo._settings import CompilationSettings

logger = logging.getLogger(__name__)

_CACHE_FORMAT_VERSION = 5
_NON_SEMANTIC_SETTINGS = {
    "cache_built_engines",
    "reuse_cached_engines",
    "cache_lowered_graphs",
    "reuse_cached_lowered_graphs",
    "dryrun",
    "lazy_engine_init",
    "timing_cache_path",
}


class BypassLoweringCache(Exception):
    """Raised when the lowered GraphModule cannot be serialized, matching Inductor's bypass."""


def _canonicalize_setting_value(value: Any) -> str:
    if isinstance(value, (set, frozenset)):
        return str(sorted(str(element) for element in value))
    return str(value)


_LITERAL_TYPES = (int, float, bool, str, type(None), bytes)


def _encode_tensor_meta(tensor: torch.Tensor) -> Dict[str, Any]:
    """Shape/dtype only. torch.empty(full_shape) allocates and made Flux 101GB."""
    from torch.fx.experimental.proxy_tensor import unset_fake_temporarily

    with unset_fake_temporarily():
        shape = []
        for dim in tensor.shape:
            if isinstance(dim, torch.SymInt):
                try:
                    shape.append(int(dim))
                except Exception:
                    shape.append(1)
            else:
                shape.append(int(dim))
        return {
            "kind": "tensor_meta",
            "shape": tuple(shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
        }


def _slim_meta_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return _encode_tensor_meta(value)
    if isinstance(value, (list, tuple)):
        return type(value)(_slim_meta_value(item) for item in value)
    return None


def _restore_meta_value(value: Any, fake_mode: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict) and value.get("kind") == "tensor_meta":
        dtype_name = value["dtype"].removeprefix("torch.")
        dtype = getattr(torch, dtype_name)
        with fake_mode:
            return torch.empty(tuple(value["shape"]), dtype=dtype, device="cpu")
    if isinstance(value, (list, tuple)):
        return type(value)(_restore_meta_value(item, fake_mode) for item in value)
    return value


def _encode_target(node: torch.fx.Node) -> Any:
    if node.op in {"placeholder", "get_attr", "call_module", "call_method", "output"}:
        if node.op == "output" and node.target is None:
            return "output"
        return node.target
    target = node.target
    if isinstance(target, torch._ops.OpOverload):
        return {"kind": "overload", "name": target.name()}
    if isinstance(target, torch._ops.OpOverloadPacket):
        return {"kind": "packet", "name": str(target)}
    if target is operator.getitem:
        return {"kind": "builtin", "name": "operator.getitem"}
    if target is getattr:
        return {"kind": "builtin", "name": "builtins.getattr"}
    module_name = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None)
    if module_name and qualname and "<" not in qualname:
        return {"kind": "import", "module": module_name, "name": qualname}
    raise BypassLoweringCache(
        f"unsupported call_function target {target!r} on node {node.name}"
    )


def _resolve_overload(name: str) -> Any:
    """Resolve 'aten::matmul', 'aten.matmul.default', or 'aten.add.Tensor'."""
    if "::" in name:
        namespace, rest = name.split("::", 1)
        if "." in rest:
            op_name, overload = rest.rsplit(".", 1)
        else:
            op_name, overload = rest, "default"
        return getattr(getattr(getattr(torch.ops, namespace), op_name), overload)
    namespace, rest = name.split(".", 1)
    if "." in rest:
        op_name, overload = rest.rsplit(".", 1)
        return getattr(getattr(getattr(torch.ops, namespace), op_name), overload)
    return getattr(getattr(torch.ops, namespace), rest)


def _decode_target(op: str, encoded: Any) -> Any:
    if op != "call_function":
        return encoded
    if not isinstance(encoded, dict):
        return encoded
    kind = encoded["kind"]
    if kind == "overload":
        return _resolve_overload(encoded["name"])
    if kind == "packet":
        name = encoded["name"]
        namespace, op_name = name.split(".", 1)
        return getattr(getattr(torch.ops, namespace), op_name)
    if kind == "builtin":
        if encoded["name"] == "operator.getitem":
            return operator.getitem
        if encoded["name"] == "builtins.getattr":
            return getattr
        raise BypassLoweringCache(f"unknown builtin {encoded['name']}")
    if kind == "import":
        module = importlib.import_module(encoded["module"])
        value: Any = module
        for part in encoded["name"].split("."):
            value = getattr(value, part)
        return value
    raise BypassLoweringCache(f"unknown target encoding {encoded}")


def _encode_arg(value: Any) -> Any:
    if isinstance(value, torch.fx.Node):
        return {"kind": "node", "name": value.name}
    if isinstance(value, _LITERAL_TYPES):
        return {"kind": "lit", "value": value}
    if isinstance(value, torch.dtype):
        return {"kind": "dtype", "value": str(value)}
    if isinstance(value, torch.device):
        return {"kind": "device", "value": str(value)}
    if isinstance(value, torch.layout):
        return {"kind": "layout", "value": str(value)}
    if isinstance(value, torch.memory_format):
        return {"kind": "memory_format", "value": str(value)}
    if isinstance(value, slice):
        return {
            "kind": "slice",
            "value": (
                _encode_arg(value.start),
                _encode_arg(value.stop),
                _encode_arg(value.step),
            ),
        }
    if value is Ellipsis:
        return {"kind": "ellipsis"}
    if isinstance(value, tuple):
        return {"kind": "tuple", "value": [_encode_arg(item) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "value": [_encode_arg(item) for item in value]}
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "value": [
                (_encode_arg(key), _encode_arg(val)) for key, val in value.items()
            ],
        }
    if isinstance(value, torch.SymInt):
        try:
            return {"kind": "lit", "value": int(value)}
        except Exception as exc:
            raise BypassLoweringCache(f"unresolvable SymInt {value}") from exc
    raise BypassLoweringCache(f"unsupported FX arg type {type(value)}: {value!r}")


def _decode_arg(encoded: Any, nodes: Dict[str, torch.fx.Node]) -> Any:
    kind = encoded["kind"]
    if kind == "node":
        return nodes[encoded["name"]]
    if kind == "lit":
        return encoded["value"]
    if kind == "dtype":
        return getattr(torch, encoded["value"].removeprefix("torch."))
    if kind == "device":
        return torch.device(encoded["value"])
    if kind == "layout":
        return getattr(torch, encoded["value"].removeprefix("torch."))
    if kind == "memory_format":
        return getattr(torch, encoded["value"].removeprefix("torch."))
    if kind == "slice":
        start, stop, step = encoded["value"]
        return slice(
            _decode_arg(start, nodes),
            _decode_arg(stop, nodes),
            _decode_arg(step, nodes),
        )
    if kind == "ellipsis":
        return Ellipsis
    if kind == "tuple":
        return tuple(_decode_arg(item, nodes) for item in encoded["value"])
    if kind == "list":
        return [_decode_arg(item, nodes) for item in encoded["value"]]
    if kind == "dict":
        return {
            _decode_arg(key, nodes): _decode_arg(val, nodes)
            for key, val in encoded["value"]
        }
    raise BypassLoweringCache(f"unknown arg encoding {encoded}")


def _assign_tensor(root: torch.nn.Module, qualified: str, tensor: torch.Tensor) -> None:
    *prefix, name = qualified.split(".")
    module = root
    for part in prefix:
        child = getattr(module, part, None)
        if child is None:
            child = torch.nn.Module()
            module.add_module(part, child)
        module = child
    tensor = tensor.detach()
    if tensor.requires_grad and tensor.is_floating_point():
        module.register_parameter(name, torch.nn.Parameter(tensor))
    else:
        module.register_buffer(name, tensor)


def _collect_state_dict(gm: torch.fx.GraphModule) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for name, tensor in gm.state_dict().items():
        if isinstance(tensor, torch.Tensor):
            state[name] = tensor
    for node in gm.graph.nodes:
        if node.op != "get_attr" or not isinstance(node.target, str):
            continue
        if node.target in state:
            continue
        try:
            value = torch.fx.graph_module._get_attr(gm, node.target)
        except Exception:
            continue
        if isinstance(value, torch.Tensor):
            state[node.target] = value
    return state


def _snapshot_node_meta(node: torch.fx.Node) -> Optional[Dict[str, Any]]:
    if "val" not in node.meta:
        return None
    return {"val": _slim_meta_value(node.meta["val"])}


class SerializedGraphModule:
    """ATen GraphModule as an explicit node list plus tensors.

    FX __reduce__ retraces generated Python and does not preserve the compiled
    graph. This encoding rebuilds torch.fx.Graph node-for-node.
    """

    def __init__(self, gm: torch.fx.GraphModule) -> None:
        self.nodes = []
        for node in gm.graph.nodes:
            self.nodes.append(
                {
                    "op": node.op,
                    "name": node.name,
                    "target": _encode_target(node),
                    "args": _encode_arg(tuple(node.args)),
                    "kwargs": _encode_arg(dict(node.kwargs)),
                    "type": node.type,
                    "meta": _snapshot_node_meta(node),
                }
            )
        self.state_dict = _collect_state_dict(gm)
        self.codegen = gm.graph._codegen
        self.in_spec = getattr(gm, "_in_spec", None)
        self.out_spec = getattr(gm, "_out_spec", None)

    def deserialize(self) -> torch.fx.GraphModule:
        from torch._subclasses.fake_tensor import FakeTensorMode

        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        root = torch.nn.Module()
        for name, tensor in self.state_dict.items():
            _assign_tensor(root, name, tensor)
        graph = torch.fx.Graph()
        created: Dict[str, torch.fx.Node] = {}
        for spec in self.nodes:
            target = _decode_target(spec["op"], spec["target"])
            args = _decode_arg(spec["args"], created)
            kwargs = _decode_arg(spec["kwargs"], created)
            node = graph.create_node(
                spec["op"],
                target,
                args,
                kwargs,
                name=spec["name"],
                type_expr=spec["type"],
            )
            if spec["meta"]:
                meta = dict(spec["meta"])
                if "val" in meta:
                    meta["val"] = _restore_meta_value(meta["val"], fake_mode)
                node.meta.update(meta)
            created[spec["name"]] = node
        gm = torch.fx.GraphModule(root, graph)
        if getattr(self, "codegen", None) is not None:
            gm.graph.set_codegen(self.codegen)
        if getattr(self, "in_spec", None) is not None:
            gm._in_spec = self.in_spec
        if getattr(self, "out_spec", None) is not None:
            gm._out_spec = self.out_spec
        gm.recompile()
        if len(list(gm.graph.nodes)) != len(self.nodes):
            raise BypassLoweringCache(
                f"rebuilt graph has {len(list(gm.graph.nodes))} nodes, saved {len(self.nodes)}"
            )
        return gm


def serialize_graph_module(gm: torch.fx.GraphModule) -> SerializedGraphModule:
    return SerializedGraphModule(gm)


def repropagate_graph_metadata(
    gm: torch.fx.GraphModule,
    arg_inputs: Sequence[Input],
    kwarg_inputs: Optional[Dict[Any, Any]],
    device: Any = "cpu",
) -> torch.fx.GraphModule:
    """Fill missing node.meta after load. No-op when the node list already restored val."""
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    if placeholders and all("val" in node.meta for node in placeholders):
        return gm
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.passes.fake_tensor_prop import FakeTensorProp
    from torch_tensorrt.dynamo.utils import get_torch_inputs, get_torch_tensor

    kwarg_map: Dict[str, Any] = {}
    if isinstance(kwarg_inputs, dict) and kwarg_inputs:
        kwarg_map = kwarg_inputs
    arg_tensors: list[Any] = []
    if arg_inputs:
        if len(arg_inputs) == 1 and isinstance(arg_inputs[0], dict) and not kwarg_map:
            kwarg_map = arg_inputs[0]
        else:
            try:
                loaded = get_torch_inputs(arg_inputs, device)
                arg_tensors = list(loaded) if not isinstance(loaded, dict) else []
            except Exception:
                arg_tensors = []
    arg_index = 0
    examples: list[Any] = []
    for node in placeholders:
        names = [node.name]
        if isinstance(node.target, str):
            names.append(node.target)
        found = None
        matched = False
        for name in names:
            if name in kwarg_map:
                found = kwarg_map[name]
                matched = True
                break
        if matched:
            if isinstance(found, Input):
                examples.append(get_torch_tensor(found, torch.device("cpu")))
            else:
                examples.append(found)
        elif arg_index < len(arg_tensors):
            examples.append(arg_tensors[arg_index])
            arg_index += 1
        else:
            examples.append(None)

    try:
        mode = FakeTensorMode(allow_non_fake_inputs=True)
        with mode:
            fake_examples = []
            for value in examples:
                if isinstance(value, torch.Tensor):
                    fake_examples.append(mode.from_tensor(value.detach().cpu()))
                else:
                    fake_examples.append(value)
            FakeTensorProp(gm, mode=mode).propagate(*fake_examples)
        logger.info("Re-propagated FakeTensor metadata on cached lowered graph")
    except Exception as exc:
        logger.warning("FakeTensorProp on cached lowered graph failed: %s", exc)
    return gm


@dataclass
class LoweringCacheEntry:
    """State needed to resume compilation after post-lowering."""

    lowered_module: torch.fx.GraphModule
    lifted_buffers: Sequence[Tuple[str, str, torch.Tensor]]


def _update_tensor_hash(hasher: Any, tensor: torch.Tensor) -> None:
    """Hash tensor metadata and contents without retaining an extra full-model copy."""
    detached = tensor.detach()
    hasher.update(str(detached.dtype).encode())
    hasher.update(str(tuple(detached.shape)).encode())
    hasher.update(str(tuple(detached.stride())).encode())
    hasher.update(str(detached.layout).encode())

    if detached.layout != torch.strided:
        detached = detached.to_dense()

    # reshape(-1) first: a 0-dim tensor cannot be viewed as uint8. Never fall
    # back to pickling the tensor, whose bytes are not content-stable across
    # processes and would make the key miss on every run.
    flat = detached.reshape(-1).contiguous().cpu()
    try:
        hasher.update(memoryview(flat.view(torch.uint8).numpy()))
    except (RuntimeError, TypeError):
        hasher.update(str(flat.tolist()).encode())


def _canonical_graph(gm: torch.fx.GraphModule) -> Tuple[str, ...]:
    return tuple(node.format_node() for node in gm.graph.nodes)


def _canonical_settings(settings: CompilationSettings) -> Tuple[Tuple[str, str], ...]:
    values = []
    for setting in fields(settings):
        if setting.name in _NON_SEMANTIC_SETTINGS:
            continue
        values.append(
            (
                setting.name,
                _canonicalize_setting_value(getattr(settings, setting.name)),
            )
        )
    return tuple(values)


def _serialize_entry(entry: LoweringCacheEntry) -> Tuple[Any, ...]:
    """Build a torch.save payload for the lowered GraphModule."""
    try:
        with torch.utils._python_dispatch._disable_current_modes():
            return (
                _CACHE_FORMAT_VERSION,
                serialize_graph_module(entry.lowered_module),
                tuple(entry.lifted_buffers),
            )
    except (
        pickle.PicklingError,
        TypeError,
        AttributeError,
        BypassLoweringCache,
    ) as exc:
        raise BypassLoweringCache(str(exc)) from exc


class DiskLoweringCache:
    """Store lowered GraphModule artifacts under a stable pre-lowering key."""

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    @staticmethod
    def can_cache(settings: CompilationSettings) -> bool:
        """Return whether the first conservative cache implementation is valid."""
        return (
            settings.require_full_compilation
            and settings.use_fast_partitioner
            and not settings.dryrun
            and not settings.enable_autocast
            and not settings.use_distributed_mode_trace
        )

    @staticmethod
    def get_hash(
        exported_program: torch.export.ExportedProgram,
        arg_inputs: Sequence[Input],
        kwarg_inputs: Optional[Dict[Any, Any]],
        settings: CompilationSettings,
    ) -> str:
        """Build a safe key available before decomposition and lowering.

        Tensor contents are included because constant folding can materialize
        weight-dependent values in the cached GraphModule.
        """
        start = time.perf_counter()
        hasher = hashlib.sha256()
        metadata = (
            _CACHE_FORMAT_VERSION,
            torch.__version__,
            torch_tensorrt_version,
            trt.__version__,
            _canonical_graph(exported_program.graph_module),
            str(exported_program.graph_signature),
            str(exported_program.range_constraints),
            tuple(str(value) for value in arg_inputs),
            tuple(
                (str(key), str(value))
                for key, value in sorted(
                    (kwarg_inputs or {}).items(), key=lambda item: str(item[0])
                )
            ),
            _canonical_settings(settings),
        )
        hasher.update(pickle.dumps(metadata))

        for name, tensor in sorted(exported_program.state_dict.items()):
            hasher.update(name.encode())
            _update_tensor_hash(hasher, tensor)

        digest = hasher.hexdigest()
        logger.info(
            "Lowering cache key computed in %.3fs (%s)",
            time.perf_counter() - start,
            digest,
        )
        return digest

    def _entry_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, key[:2], key)

    def load(self, key: str) -> Optional[LoweringCacheEntry]:
        directory = self._entry_path(key)
        artifact_path = os.path.join(directory, "lowered.pt")
        if not os.path.exists(artifact_path):
            logger.info("Lowering cache miss for key %s", key)
            return None
        try:
            version, serialized_gm, lifted_buffers = torch.load(
                artifact_path, map_location="cpu", weights_only=False
            )
            if version != _CACHE_FORMAT_VERSION:
                raise BypassLoweringCache(
                    f"cache format {version} != {_CACHE_FORMAT_VERSION}"
                )
            if not isinstance(serialized_gm, SerializedGraphModule):
                raise BypassLoweringCache(
                    f"expected SerializedGraphModule, got {type(serialized_gm)}"
                )
            lowered_module = serialized_gm.deserialize()
            lowered_module.recompile()
        except Exception as exc:
            logger.warning(
                "Ignoring unreadable lowering cache entry %s: %s", directory, exc
            )
            return None
        os.utime(artifact_path, None)
        logger.info("Lowering cache hit for key %s", key)
        return LoweringCacheEntry(lowered_module, tuple(lifted_buffers))

    def save(self, key: str, entry: LoweringCacheEntry) -> torch.fx.GraphModule:
        directory = self._entry_path(key)
        os.makedirs(directory, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(
            prefix="lowered-", suffix=".pt", dir=directory
        )
        os.close(fd)
        try:
            payload = _serialize_entry(entry)
            with torch.utils._python_dispatch._disable_current_modes():
                torch.save(payload, temporary_path)
            os.replace(temporary_path, os.path.join(directory, "lowered.pt"))
            logger.info("Saved lowering cache entry for key %s", key)
        except BypassLoweringCache as exc:
            logger.warning("Bypassing lowering cache save for %s: %s", directory, exc)
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
        except Exception as exc:
            logger.warning("Failed to save lowering cache entry %s: %s", directory, exc)
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
        return entry.lowered_module
