from __future__ import annotations

import json
from inspect import Parameter, signature
from pathlib import Path
from typing import Any

import torch
import torch_tensorrt

from .measure import cuda_ms
from .ops import _as_tuple, record_engine
from .spec import ComponentBundle

DEFAULT_TRT_SETTINGS: dict[str, Any] = {
    "min_block_size": 1,
    "require_full_compilation": True,
    "immutable_weights": True,
    "disable_tf32": True,
    "truncate_double": True,
}

_TRT_COMPILE_KEYS = frozenset(DEFAULT_TRT_SETTINGS) | {
    "use_fp32_acc",
    "truncate_double",
    "decompose_attention",
    "offload_module_to_cpu",
    "assume_dynamic_shape_support",
    "use_explicit_typing",
}


def compile_component(
    bundle: ComponentBundle,
    *,
    name: str,
    engine_dir: Path,
    trt_settings: dict[str, Any] | None = None,
) -> tuple[str, tuple[torch.Tensor, ...], float]:
    """Export one component, compile it, write ``engine_dir/<name>/``.

    Family setattr is owned by ``EdgeSpec.apply_patches`` around this call.
    ``execute_engine`` records the TensorRT module, not eager.
    """
    from .plugin.attn_patches import (
        set_language_mask_type,
    )

    module = bundle.module.eval()
    trace_args = tuple(bundle.trace_args)
    save_args = tuple(bundle.save_args)
    execute_args = tuple(bundle.execute_args or save_args)
    output_subdir = name if bundle.output_subdir is None else bundle.output_subdir
    out_dir = Path(engine_dir) / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    engine_path = str(out_dir)

    if bundle.context_attention_mask_type is not None:
        set_language_mask_type(bundle.context_attention_mask_type)

    export_kwargs: dict[str, Any] = {"strict": False}
    if bundle.input_specs is not None:
        from torch_tensorrt.dynamo._tracer import build_dim_registry, get_dynamic_shapes

        specs = tuple(bundle.input_specs)
        leading = 0
        for input_name in bundle.input_names:
            if input_name.startswith("past_key_values"):
                break
            leading += 1
        dim_registry = build_dim_registry(specs[:leading], {})
        dynamic_shapes: dict[str, Any] = {}
        positional_names: list[str] = []
        var_pos_name: str | None = None
        for param in signature(module.forward).parameters.values():
            if param.kind == Parameter.VAR_POSITIONAL:
                var_pos_name = param.name
                break
            if param.kind in (
                Parameter.POSITIONAL_ONLY,
                Parameter.POSITIONAL_OR_KEYWORD,
            ):
                positional_names.append(param.name)
        for spec, param_name in zip(specs[:leading], positional_names[:leading]):
            if bundle.edge_runtime_bindings and param_name == "kvcache_start_index":
                dynamic_shapes[param_name] = {}
            elif bundle.edge_runtime_bindings or param_name in (
                "inputs_embeds",
                "ds_stack",
            ):
                dynamic_shapes[param_name] = get_dynamic_shapes(spec, dim_registry)
            else:
                dynamic_shapes[param_name] = {}
        if var_pos_name is not None:
            dynamic_shapes[var_pos_name] = tuple(
                get_dynamic_shapes(spec, dim_registry) for spec in specs[leading:]
            )
        export_kwargs["dynamic_shapes"] = dynamic_shapes

    exported = torch.export.export(module, args=trace_args, **export_kwargs)
    settings = {
        k: v
        for k, v in {
            **DEFAULT_TRT_SETTINGS,
            **(trt_settings or {}),
            **bundle.trt_settings,
        }.items()
        if k in _TRT_COMPILE_KEYS
    }

    arg_inputs = (
        tuple(bundle.input_specs) if bundle.input_specs is not None else trace_args
    )
    engine_file = bundle.engine_file
    if bundle.edge_runtime_bindings:
        serialized = (
            torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
                exported,
                arg_inputs=arg_inputs,
                arg_input_binding_names=tuple(bundle.input_names),
                output_binding_names=tuple(bundle.output_names),
                **settings,
            )
        )
        (out_dir / engine_file).write_bytes(serialized)
        aliased_io = {}
        for output_name in bundle.output_names:
            if output_name.startswith("present_key_values_"):
                layer_index = output_name.rsplit("_", 1)[-1]
                aliased_io[output_name] = (
                    f"past_key_values_{layer_index}",
                    "kv_cache_update",
                )
            elif output_name.startswith("present_k_cache_"):
                layer_index = output_name.rsplit("_", 1)[-1]
                aliased_io[output_name] = (
                    f"k_cache_{layer_index}",
                    "kv_cache_update",
                )
            elif output_name.startswith("present_v_cache_"):
                layer_index = output_name.rsplit("_", 1)[-1]
                aliased_io[output_name] = (
                    f"v_cache_{layer_index}",
                    "kv_cache_update",
                )

        from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

        compiled = TorchTensorRTModule(
            serialized_engine=serialized,
            input_binding_names=list(bundle.input_names),
            output_binding_names=list(bundle.output_names),
            name=name,
            aliased_io=aliased_io,
        )
        with torch.no_grad():
            trt_out = tuple(
                tensor.detach().clone() for tensor in _as_tuple(compiled(*execute_args))
            )
        trt_ms = cuda_ms(lambda: compiled(*execute_args))
        record_engine(
            engine_path,
            component=name,
            input_names=bundle.input_names,
            outputs=trt_out,
            module=compiled,
        )
        _write_sidecar(
            out_dir,
            bundle,
            name,
            trt_out,
            engine_file=engine_file,
        )
        if bundle.artifact_writer is not None:
            bundle.artifact_writer(out_dir)
        return engine_path, trt_out, trt_ms

    compiled = torch_tensorrt.dynamo.compile(
        exported,
        arg_inputs=arg_inputs,
        **settings,
    )

    with torch.no_grad():
        trt_out = tuple(
            tensor.detach().clone() for tensor in _as_tuple(compiled(*execute_args))
        )
    trt_ms = cuda_ms(lambda: compiled(*execute_args))

    record_engine(
        engine_path,
        component=name,
        input_names=bundle.input_names,
        outputs=trt_out,
        module=compiled,
    )
    # ``dynamo.compile`` has already built and serialized the engine. Reuse
    # those bytes instead of calling
    # ``convert_exported_program_to_serialized_trt_engine`` and building the
    # same engine a second time, which can exceed GPU memory for large models.
    serialized_engines = [
        submodule.serialized_engine
        for submodule in compiled.modules()
        if getattr(submodule, "serialized_engine", None) is not None
    ]
    if len(serialized_engines) == 1:
        serialized = bytes(serialized_engines[0])
    elif not serialized_engines:
        # Test doubles and alternative compiler backends may not expose the
        # runtime module. Preserve the public serialization API as a fallback.
        serialized = (
            torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
                exported,
                arg_inputs=arg_inputs,
                **settings,
            )
        )
    else:
        raise RuntimeError(
            f"Expected one fully compiled TensorRT engine for {name}, "
            f"found {len(serialized_engines)}"
        )
    (out_dir / engine_file).write_bytes(serialized)

    _write_sidecar(out_dir, bundle, name, trt_out, engine_file=engine_file)
    if bundle.artifact_writer is not None:
        bundle.artifact_writer(out_dir)
    return engine_path, trt_out, trt_ms


def _write_sidecar(
    out_dir: Path,
    bundle: ComponentBundle,
    name: str,
    outputs: tuple[torch.Tensor, ...],
    *,
    engine_file: str | None = None,
) -> None:
    config = {
        "model_type": bundle.model_type,
        "component": name,
        "engine_file": engine_file or bundle.engine_file,
        "input_names": list(bundle.input_names),
        "output_names": list(bundle.output_names),
        "outputs": [{"shape": list(t.shape), "dtype": str(t.dtype)} for t in outputs],
    }
    config.update(bundle.extra_config)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
