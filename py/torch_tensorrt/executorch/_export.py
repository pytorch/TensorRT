from __future__ import annotations

import logging
import platform
from collections.abc import Mapping, Sequence
from inspect import signature
from typing import TYPE_CHECKING, Any

import torch
from torch.export import ExportedProgram
from torch_tensorrt._Input import Input

if TYPE_CHECKING:
    from executorch.exir import EdgeCompileConfig, EdgeProgramManager
    from executorch.exir.backend.compile_spec_schema import CompileSpec
    from executorch.exir.backend.partitioner import Partitioner

logger = logging.getLogger(__name__)


def _contains_dynamic_input(value: Any) -> bool:
    if isinstance(value, Input):
        return bool(value.shape_mode == Input._ShapeMode.DYNAMIC)
    if isinstance(value, Mapping):
        return any(_contains_dynamic_input(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_dynamic_input(item) for item in value)
    return False


def _all_input_specs(value: Any) -> bool:
    if isinstance(value, Input):
        return True
    if isinstance(value, Mapping):
        return all(_all_input_specs(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_all_input_specs(item) for item in value)
    return False


def _dynamic_shapes_from_input(value: Any, dim_registry: dict[str, Any]) -> Any:
    from torch_tensorrt.dynamo._tracer import get_dynamic_shapes

    if isinstance(value, Input):
        return get_dynamic_shapes(value, dim_registry)
    if isinstance(value, Mapping):
        return {
            key: _dynamic_shapes_from_input(item, dim_registry)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_dynamic_shapes_from_input(item, dim_registry) for item in value)
    if isinstance(value, list):
        return [_dynamic_shapes_from_input(item, dim_registry) for item in value]
    raise TypeError(f"Unsupported input structure leaf: {type(value).__name__}")


def _to_torch_inputs(value: Any, device: torch.device) -> Any:
    from torch_tensorrt.dynamo.utils import get_torch_tensor

    if isinstance(value, Input):
        return get_torch_tensor(value, device)
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _to_torch_inputs(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_torch_inputs(item, device) for item in value)
    if isinstance(value, list):
        return [_to_torch_inputs(item, device) for item in value]
    return value


def _prepare_graph_module(
    module: torch.fx.GraphModule,
    *,
    arg_inputs: Sequence[Any] | None,
    kwarg_inputs: Mapping[str, Any] | None,
    dynamic_shapes: Any | None,
    retrace: bool,
) -> ExportedProgram:
    from torch_tensorrt.dynamo._defaults import default_device
    from torch_tensorrt.dynamo._exporter import export as export_graph_module
    from torch_tensorrt.dynamo._tracer import build_dim_registry
    from torch_tensorrt.dynamo.utils import to_torch_device

    arguments = tuple(arg_inputs or ())
    keyword_arguments = dict(kwarg_inputs or {})
    all_specs = (
        bool(arguments or keyword_arguments)
        and _all_input_specs(arguments)
        and _all_input_specs(keyword_arguments)
    )
    if dynamic_shapes is None and all_specs:
        bound_arguments = (
            signature(module.forward).bind(*arguments, **keyword_arguments).arguments
        )
        dim_registry = build_dim_registry(arguments, keyword_arguments)
        dynamic_shapes = {
            name: _dynamic_shapes_from_input(value, dim_registry)
            for name, value in bound_arguments.items()
        }
    elif dynamic_shapes is None and (
        _contains_dynamic_input(arguments) or _contains_dynamic_input(keyword_arguments)
    ):
        raise ValueError(
            "Mixed Tensor and dynamic Input arguments require explicit dynamic_shapes."
        )

    device = to_torch_device(default_device())
    torch_arguments = tuple(_to_torch_inputs(arguments, device))
    torch_keyword_arguments = _to_torch_inputs(keyword_arguments, device)
    if retrace and not torch_arguments and not torch_keyword_arguments:
        raise ValueError("retrace=True requires example inputs.")

    return export_graph_module(
        module,
        arg_inputs=torch_arguments,
        kwarg_inputs=torch_keyword_arguments,
        dynamic_shapes=dynamic_shapes,
        use_legacy_exporter=not retrace,
    )


def _copyback_buffers_by_method(
    source: Any, method_names: tuple[str, ...]
) -> dict[str, list[str]]:
    """Per-method copy-back buffer names, read from the source's ``meta``.

    ``compile()`` records the flattened names of the non-KV mutable buffers that
    need a write-back copy in ``gm.meta["_copyback_mutation_buffers"]``.
    """

    def _names(obj: Any) -> list[str]:
        gm = (
            obj
            if isinstance(obj, torch.fx.GraphModule)
            else getattr(obj, "graph_module", None)
        )
        meta = getattr(gm, "meta", {}) or {}
        return list(meta.get("_copyback_mutation_buffers", []) or [])

    if isinstance(source, Mapping):
        return {name: _names(source[name]) for name in method_names}
    return {method_names[0]: _names(source)}


def _prepare_programs(
    source: ExportedProgram | torch.fx.GraphModule | Mapping[str, ExportedProgram],
    *,
    arg_inputs: Sequence[Any] | None,
    kwarg_inputs: Mapping[str, Any] | None,
    dynamic_shapes: Any | None,
    retrace: bool | None,
) -> tuple[ExportedProgram | dict[str, ExportedProgram], tuple[str, ...]]:
    """Normalize the source into the programs to export and their method names.

    Rejecting one ExportedProgram object under two method names is this module's rule,
    not ExecuTorch's, which accepts it. Every method is staged and rewritten on its own,
    so passing one object twice shares nothing and is almost always a mistake.
    """
    source_options_used = (
        arg_inputs is not None
        or kwarg_inputs is not None
        or dynamic_shapes is not None
        or retrace is True
    )
    if isinstance(source, ExportedProgram):
        if source_options_used:
            raise ValueError(
                "Inputs, dynamic_shapes, and retrace=True are invalid for an "
                "already-exported program."
            )
        return source, ("forward",)

    if isinstance(source, torch.fx.GraphModule):
        program = _prepare_graph_module(
            source,
            arg_inputs=arg_inputs,
            kwarg_inputs=kwarg_inputs,
            dynamic_shapes=dynamic_shapes,
            retrace=False if retrace is None else retrace,
        )
        return program, ("forward",)

    if isinstance(source, Mapping):
        if not source:
            raise ValueError("Method mapping must not be empty.")
        if source_options_used:
            raise ValueError(
                "Method mappings accept only pre-exported programs; per-method "
                "input and retrace options are not supported."
            )
        programs: dict[str, ExportedProgram] = {}
        program_names_by_id: dict[int, str] = {}
        for name, program in source.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Every method name must be a non-empty string.")
            if not isinstance(program, ExportedProgram):
                raise TypeError(
                    "Method mappings must contain only ExportedProgram values."
                )
            previous_name = program_names_by_id.get(id(program))
            if previous_name is not None:
                raise ValueError(
                    "Method mapping contains the same ExportedProgram object for "
                    f"{previous_name!r} and {name!r}. Each method requires an "
                    "independent program."
                )
            program_names_by_id[id(program)] = name
            programs[name] = program
        return programs, tuple(programs)

    raise TypeError(
        "source must be a TensorRT-compiled GraphModule, an engine-bearing "
        "ExportedProgram, or a mapping of method names to ExportedPrograms. "
        "Compile nn.Module inputs with torch_tensorrt.compile() first."
    )


def _reject_misnamed_partitioners(per_method: dict[str, list[Any]]) -> None:
    """Reject a partitioner whose compile specs name a method other than its own.

    A partitioner holds its compile specs from construction, so the name in those specs
    is the name its delegates carry. When that is not the method the partitioner was
    given, the delegates look up the wrong compiled method at runtime. Sharing one
    named instance across methods is the common way to hit this, since at most one of
    those methods can match.

    An instance whose specs carry no method name is not rejected, because some backends
    are built to share one across methods. A backend that instead reads its method name
    from its specs raises during ExecuTorch's own lowering, so this is not a case that
    can be decided here.
    """
    for name, partitioners in per_method.items():
        for partitioner in partitioners:
            declared = _declared_method_name(partitioner)
            if declared is not None and declared != name:
                raise ValueError(
                    f"partitioners[{name!r}] holds a {type(partitioner).__name__} whose "
                    f"compile specs name method {declared!r}. The delegate would be "
                    f"labelled {declared!r} and look up the wrong compiled method at "
                    f"runtime. Build it with the spec for {name!r}."
                )


def _declared_method_name(partitioner: Any) -> str | None:
    """The method a partitioner's compile specs name, if they name one.

    Only the specs the partitioner holds from construction are visible here. A
    partitioner that builds its DelegationSpec inside partition() is not detected.
    """
    spec = getattr(partitioner, "delegation_spec", None)
    for compile_spec in getattr(spec, "compile_specs", None) or ():
        if getattr(compile_spec, "key", None) != "method_name":
            continue
        value = getattr(compile_spec, "value", None)
        if isinstance(value, (bytes, bytearray)):
            return value.decode()
        return None if value is None else str(value)
    return None


def _apply_weight_streaming_budget(
    method_compile_specs: dict[str, list[Any]],
    weight_streaming_budget_per_engine: Any,
) -> None:
    """Bake the per-engine weight streaming budget into every method's compile specs.

    The raw compile spec is rejected even when no budget is given, because writing it
    by hand skips the validation here and silently disagrees with a budget passed the
    supported way.
    """
    from executorch.exir.backend.compile_spec_schema import CompileSpec
    from torch_tensorrt.executorch.partitioner import (
        normalize_weight_streaming_budget_per_engine,
        WEIGHT_STREAMING_BUDGET_COMPILE_SPEC_KEY,
    )

    for name, specs in method_compile_specs.items():
        if any(
            getattr(spec, "key", None) == WEIGHT_STREAMING_BUDGET_COMPILE_SPEC_KEY
            for spec in specs
        ):
            raise ValueError(
                f"compile_specs for {name!r} carries a "
                f"CompileSpec({WEIGHT_STREAMING_BUDGET_COMPILE_SPEC_KEY!r}, ...). Pass "
                "weight_streaming_budget_per_engine instead."
            )
    spec_value = normalize_weight_streaming_budget_per_engine(
        weight_streaming_budget_per_engine
    )
    if spec_value is None:
        return
    for specs in method_compile_specs.values():
        specs.append(CompileSpec(WEIGHT_STREAMING_BUDGET_COMPILE_SPEC_KEY, spec_value))


def _per_method_values(
    value: Sequence[Any] | Mapping[str, Sequence[Any] | None] | None,
    method_names: tuple[str, ...],
    option_name: str,
) -> dict[str, list[Any]]:
    if isinstance(value, Mapping):
        unknown = set(value) - set(method_names)
        if unknown:
            raise ValueError(
                f"{option_name} contains unknown methods: {sorted(unknown)}"
            )
        normalized: dict[str, list[Any]] = {}
        for name in method_names:
            method_value = value.get(name)
            if method_value is not None and (
                not isinstance(method_value, Sequence)
                or isinstance(method_value, (str, bytes))
            ):
                raise TypeError(f"{option_name}[{name!r}] must be a sequence or None.")
            normalized[name] = list(method_value or ())
        return normalized
    if value is not None and (
        not isinstance(value, Sequence) or isinstance(value, (str, bytes))
    ):
        raise TypeError(
            f"{option_name} must be a sequence, or a mapping of method name to "
            "sequence."
        )
    shared = list(value or ())
    return {name: list(shared) for name in method_names}


def export(
    source: ExportedProgram | torch.fx.GraphModule | Mapping[str, ExportedProgram],
    *,
    inputs: Sequence[Any] | None = None,
    arg_inputs: Sequence[Any] | None = None,
    kwarg_inputs: Mapping[str, Any] | None = None,
    dynamic_shapes: Any | None = None,
    retrace: bool | None = None,
    transform_passes: Any | None = None,
    partitioners: (
        Sequence["Partitioner"] | Mapping[str, Sequence["Partitioner"] | None] | None
    ) = None,
    compile_specs: (
        Sequence["CompileSpec"] | Mapping[str, Sequence["CompileSpec"] | None] | None
    ) = None,
    compile_config: "EdgeCompileConfig | None" = None,
    constant_methods: Mapping[str, Any] | None = None,
    generate_etrecord: bool = False,
    weight_streaming_budget_per_engine: int | None = None,
) -> "EdgeProgramManager":
    """Prepare TensorRT-compiled programs for composable ExecuTorch lowering.

    TensorRT claims engine nodes first. Additional partitioners run afterward in
    caller-provided order. The returned EdgeProgramManager is the standard
    ExecuTorch inspection and customization boundary; call ``to_executorch()``
    on it when ready to perform final memory planning and serialization.

    Export stages independent graph and metadata containers while sharing tensor
    payload storage with the source programs, so transform passes must treat a
    shared payload value as immutable. Engines are never deep copied, since that
    would serialize and deserialize them, but each one is decoded into a byte
    buffer the returned program owns, so the bytes of a multi-gigabyte engine are
    resident twice while both programs are alive. Method mappings preserve
    independent entry points but do not imply shared mutable state between them.

    When exporting more than one method, give each method its own partitioner
    instances via ``partitioners={"method": [...]}``. A partitioner may carry
    method-specific state. Give each instance the compile spec for the method it serves,
    since a backend that reads its method name from the specs cannot find it otherwise.
    Sharing one instance across methods is rejected when its specs name a method, because
    every method sharing it would be tagged with the same name. Sharing an instance whose
    specs name no method is not rejected here, but a backend that reads its own method
    name from its specs, such as the CUDA backend, then raises during lowering.

    ``generate_etrecord=True`` is outside the payload sharing described above. It makes
    ExecuTorch deep copy the whole program, so peak memory grows by roughly the size of
    the program including engines.

    Each engine is serialized once, into base64 text. Every method is serialized before
    the first one is rewritten, and a method's text is released once that method is
    rewritten. On top of the engine bytes the exported program itself carries, peak
    memory adds at most roughly 1.33x the engine bytes of every method together, which
    is the base64 text, plus about 2.3x the bytes of the largest engine, which is what
    decoding one method's text holds at once. For methods whose engines are the same
    size that is about 3.7x for one method, 2.5x for two, and 1.9x for four.

    ``constant_methods`` keys are restricted to valid Python identifiers here, which is
    narrower than ExecuTorch itself accepts.

    Arguments:
        source (Union(torch.export.ExportedProgram, torch.fx.GraphModule, Dict[str, torch.export.ExportedProgram])):
            A TensorRT-compiled source. A GraphModule is exported first, and a mapping
            becomes one method per key. An ExportedProgram and a mapping both need their
            engines already compiled, so ``inputs``, ``arg_inputs``, ``kwarg_inputs``,
            ``dynamic_shapes`` and ``retrace=True`` are rejected for them.

    Keyword Arguments:
        inputs (Sequence[Union(torch_tensorrt.Input, torch.Tensor)]): Example positional
            inputs used to export a GraphModule source. Mutually exclusive with
            ``arg_inputs``, which is the same argument under the newer name.
        arg_inputs (Sequence[Union(torch_tensorrt.Input, torch.Tensor)]): See ``inputs``.
        kwarg_inputs (Dict[str, Any]): Example keyword inputs used to export a
            GraphModule source.
        dynamic_shapes (Any): Dynamic shape specification passed through to
            ``torch.export``. Inferred from any ``torch_tensorrt.Input`` in the example
            inputs when omitted, which requires every input to be an ``Input``.
        retrace (bool): Re-trace a GraphModule source instead of wrapping its existing
            graph. Unset behaves as False, and True requires example inputs.
        transform_passes (Union(Sequence[Any], Dict[str, Sequence[Any]], PassManager)):
            ExecuTorch transform passes, either for every method or per method.
        partitioners (Union(Sequence[Partitioner], Dict[str, Sequence[Partitioner]])):
            Extra partitioners to run after the TensorRT one, either for every method or
            per method. Give each method its own instances.
        compile_specs (Union(Sequence[CompileSpec], Dict[str, Sequence[CompileSpec]])):
            Compile specs for the TensorRT partitioner, either for every method or per
            method.
        compile_config (executorch.exir.EdgeCompileConfig): Edge compile config.
            Defaults to :func:`get_edge_compile_config`.
        constant_methods (Dict[str, Any]): Methods returning a constant, such as a vocab
            size. Keys must be valid Python identifiers and must not name a method of
            ``source``.
        generate_etrecord (bool): Ask ExecuTorch for an ETRecord for later debugging.
            This copies the whole program, engines included.
        weight_streaming_budget_per_engine (Optional[int]): Bytes of engine weights that
            may stay resident in GPU memory, with the rest streamed from host memory.
            It applies to **each** TensorRT engine separately, not as a total for the
            program, so a program with N engines can hold up to N times this value
            resident: every delegate is initialized when its method loads and they stay
            resident together. Requires the engine to have been compiled with
            ``enable_weight_streaming=True``; it is ignored, with a log message, on an
            engine that cannot stream. Leave it unset, the default, so each engine gets
            TensorRT's own automatic budget, sized against free memory on the device it
            actually loads on. This is a different unit from
            ``torch_tensorrt.runtime.weight_streaming(...).device_budget``, which is a
            program total split proportionally across engines.

    Returns:
        executorch.exir.EdgeProgramManager: The Edge program, ready for inspection,
        further transformation, or ``to_executorch()``.
    """
    from torch_tensorrt._features import ENABLED_FEATURES

    if platform.system() != "Linux":
        raise ValueError(
            f"The executorch format is only supported on Linux, {platform.system()} "
            "is not a supported platform for this format"
        )
    if not ENABLED_FEATURES.torch_tensorrt_runtime:
        raise RuntimeError(
            "ExecuTorch export requires the Torch-TensorRT runtime "
            "(torch_tensorrt_runtime). Reinstall torch_tensorrt with the runtime extension."
        )
    if inputs is not None and arg_inputs is not None:
        raise ValueError("inputs and arg_inputs are mutually exclusive.")
    arguments = inputs if inputs is not None else arg_inputs

    import torch_tensorrt.dynamo.runtime.meta_ops.register_meta_ops  # noqa: F401
    from executorch.exir import to_edge_transform_and_lower
    from torch_tensorrt.dynamo._exporter import _declare_aliased_kv_mutations_on_ep
    from torch_tensorrt.executorch import TensorRTPartitioner, get_edge_compile_config
    from torch_tensorrt.executorch._export_utils import (
        replace_execute_engine,
        stage_exported_program,
        validate_engine_program,
    )

    programs, method_names = _prepare_programs(
        source,
        arg_inputs=arguments,
        kwarg_inputs=kwarg_inputs,
        dynamic_shapes=dynamic_shapes,
        retrace=retrace,
    )
    program_map = {"forward": programs} if not isinstance(programs, dict) else programs
    copyback_by_method = _copyback_buffers_by_method(source, method_names)
    extra_partitioners = _per_method_values(partitioners, method_names, "partitioners")
    _reject_misnamed_partitioners(extra_partitioners)
    method_compile_specs = _per_method_values(
        compile_specs, method_names, "compile_specs"
    )
    _apply_weight_streaming_budget(
        method_compile_specs, weight_streaming_budget_per_engine
    )

    if constant_methods is not None:
        invalid = [
            name
            for name in constant_methods
            if not isinstance(name, str) or not name.isidentifier()
        ]
        if invalid:
            raise ValueError(
                f"constant_methods keys must be valid Python identifiers: {invalid}"
            )
        collisions = set(constant_methods) & set(method_names)
        if collisions:
            raise ValueError(
                f"constant_methods collide with executable methods: {sorted(collisions)}"
            )

    if isinstance(transform_passes, Mapping):
        unknown = set(transform_passes) - set(method_names)
        if unknown:
            raise ValueError(
                f"transform_passes contains unknown methods: {sorted(unknown)}"
            )
        # ExecuTorch dispatches per-method passes on isinstance(passes, dict), and a
        # Mapping that is not a dict matches none of its branches, so it reaches a
        # KeyError on the first method. An empty dict fails the same way, so treat it
        # as no passes at all. Every method the dict omits is deep-copied instead,
        # which copies that method's whole engine buffer, so give the omitted ones an
        # empty pass list: it runs nothing and hands back the same program.
        transform_passes = (
            {name: list(transform_passes.get(name, ())) for name in method_names}
            if transform_passes
            else None
        )

    resolved_engines: dict[str, dict[str, list[Any]]] = {
        name: {} for name in program_map
    }
    engine_call_counts = {
        name: validate_engine_program(program, resolved_engines[name])
        for name, program in program_map.items()
    }
    # Zero-engine methods are allowed: later partitioners may claim their ops,
    # or portable operators may remain undelegated.

    # An undeclared copy-back mutation reaches ExecuTorch as a trailing user output that
    # nothing copies back, and an undeclared engine-aliased KV write does not reach it
    # at all, since torch.export drops those outputs at the fx boundary. Either way the
    # buffer loads frozen at its serialized value and never updates. Only the
    # legacy exporter declares them while building the program, so a retraced program
    # always needs the declaration and a caller's pre-exported one may or may not; the
    # pass is idempotent, so every source shape can go through it. It rewrites the
    # graph's output node in place while returning the rewritten signature on a new
    # ExportedProgram, so it runs on the staged copy: given the caller's own program it
    # would leave that program's graph and signature describing different outputs, which
    # only the ExportedProgram verifier -- reached through save() -- reports. Staging
    # holds engines and weights by reference, so declaring here costs no extra copy.
    staged_programs = {
        name: _declare_aliased_kv_mutations_on_ep(
            stage_exported_program(program),
            copyback_buffers=copyback_by_method.get(name, []),
        )
        for name, program in program_map.items()
    }
    rewritten: dict[str, ExportedProgram] = {}
    method_partitioners: dict[str, list[Partitioner]] = {}
    for name, program in staged_programs.items():
        if engine_call_counts[name] > 1:
            logger.warning(
                "%s contains %d TRT engine calls. Each one becomes its own delegate, "
                "so the boundary overhead is paid that many times.",
                name,
                engine_call_counts[name],
            )
            if weight_streaming_budget_per_engine is not None:
                logger.warning(
                    "weight_streaming_budget_per_engine applies to each of those %d "
                    "engines separately, not as a total, so resident weights for %s "
                    "can reach %d times the value given.",
                    engine_call_counts[name],
                    name,
                    engine_call_counts[name],
                )
        # Drop this method's engine payloads as soon as they are in the graph.
        rewritten[name] = replace_execute_engine(program, resolved_engines.pop(name))
        method_partitioners[name] = [
            TensorRTPartitioner(compile_specs=method_compile_specs[name]),
            *extra_partitioners[name],
        ]

    edge_programs: ExportedProgram | dict[str, ExportedProgram]
    partitioner_pipeline: list[Partitioner] | dict[str, list[Partitioner]]
    if isinstance(programs, dict):
        edge_programs = rewritten
        partitioner_pipeline = method_partitioners
    else:
        edge_programs = rewritten["forward"]
        partitioner_pipeline = method_partitioners["forward"]

    return to_edge_transform_and_lower(
        edge_programs,
        transform_passes=transform_passes,
        partitioner=partitioner_pipeline,
        constant_methods=dict(constant_methods) if constant_methods else None,
        compile_config=(
            compile_config if compile_config is not None else get_edge_compile_config()
        ),
        generate_etrecord=generate_etrecord,
    )
