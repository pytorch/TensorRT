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


def _reject_shared_method_named_partitioners(per_method: dict[str, list[Any]]) -> None:
    """Reject one partitioner instance carrying a method name serving several methods.

    A partitioner holds its compile specs from construction. When those specs name a
    method, every method sharing that instance is tagged with the same name, and the
    delegates then look up the wrong compiled method at runtime. Sharing an instance
    whose specs carry no method name is fine, and ExecuTorch's own multi-method examples
    do it, so only the naming case is an error.
    """
    if len(per_method) < 2:
        return

    owner_by_id: dict[int, str] = {}
    for name, partitioners in per_method.items():
        for partitioner in partitioners:
            if not _carries_method_name(partitioner):
                continue
            previous = owner_by_id.setdefault(id(partitioner), name)
            if previous != name:
                raise ValueError(
                    f"partitioners reuses the same {type(partitioner).__name__} "
                    f"instance for {previous!r} and {name!r}, and its compile specs "
                    "name a method, so both methods would be tagged with the same "
                    "name. Give each method its own instance: "
                    'partitioners={"prefill": [MyPartitioner(...)], "decode": '
                    "[MyPartitioner(...)]}."
                )


def _carries_method_name(partitioner: Any) -> bool:
    """Whether a partitioner's compile specs name a specific method.

    Only the specs the partitioner holds from construction are visible here. A
    partitioner that builds its DelegationSpec inside partition() is not detected.
    """
    spec = getattr(partitioner, "delegation_spec", None)
    for compile_spec in getattr(spec, "compile_specs", None) or ():
        if getattr(compile_spec, "key", None) == "method_name":
            return True
    return False


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
) -> "EdgeProgramManager":
    """Prepare TensorRT-compiled programs for composable ExecuTorch lowering.

    TensorRT claims engine nodes first. Additional partitioners run afterward in
    caller-provided order. The returned EdgeProgramManager is the standard
    ExecuTorch inspection and customization boundary; call ``to_executorch()``
    on it when ready to perform final memory planning and serialization.

    Export stages independent graph and metadata containers while sharing tensor
    and engine payload storage, avoiding copies of potentially multi-gigabyte
    TensorRT engines. Transform passes must treat shared payload values as
    immutable. Method mappings preserve independent entry points but do not imply
    shared mutable state between them.

    When exporting more than one method, give each method its own partitioner
    instances via ``partitioners={"method": [...]}``. A partitioner may carry
    method-specific state. Give each instance the compile spec for the method it serves,
    since a backend that reads its method name from the specs cannot find it otherwise.
    Sharing one instance across methods is rejected when its specs name a method, because
    every method sharing it would be tagged with the same name. Sharing an instance whose
    specs name no method is allowed.

    ``generate_etrecord=True`` is outside the payload sharing described above. It makes
    ExecuTorch deep copy the whole program, so peak memory grows by roughly the size of
    the program including engines.

    Each engine is serialized once, and a method's engine payloads are released as soon
    as that method is rewritten. While a method is being rewritten, its engines are
    resident both as the base64 text the serialization produced and as the decoded
    buffer, so peak memory is roughly 2.3x that method's engine bytes, and briefly 3.3x
    for the engine being decoded.

    ``constant_methods`` keys are restricted to valid Python identifiers here, which is
    narrower than ExecuTorch itself accepts.
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
    extra_partitioners = _per_method_values(partitioners, method_names, "partitioners")
    _reject_shared_method_named_partitioners(extra_partitioners)
    method_compile_specs = _per_method_values(
        compile_specs, method_names, "compile_specs"
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
    engine_counts = {
        name: validate_engine_program(program, resolved_engines[name])
        for name, program in program_map.items()
    }
    # Zero-engine methods are allowed: later partitioners may claim their ops,
    # or portable operators may remain undelegated.
    staged_programs = {
        name: stage_exported_program(program) for name, program in program_map.items()
    }
    rewritten: dict[str, ExportedProgram] = {}
    method_partitioners: dict[str, list[Partitioner]] = {}
    for name, program in staged_programs.items():
        if engine_counts[name] > 1:
            logger.warning(
                "%s contains %d TRT engines. Multi-engine .pte exports can incur "
                "extra delegate boundary overhead.",
                name,
                engine_counts[name],
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
