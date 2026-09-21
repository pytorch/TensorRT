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
    out_dir = Path(engine_dir) / name
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
            if param_name in ("inputs_embeds", "ds_stack"):
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
    compiled = torch_tensorrt.dynamo.compile(
        exported,
        arg_inputs=arg_inputs,
        **settings,
    )

    with torch.no_grad():
        trt_out = _as_tuple(compiled(*execute_args))
    trt_ms = cuda_ms(lambda: compiled(*execute_args))

    record_engine(
        engine_path,
        component=name,
        input_names=bundle.input_names,
        outputs=trt_out,
        module=compiled,
    )
    engine_file = bundle.engine_file

    serialized = (
        torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
            exported,
            arg_inputs=arg_inputs,
            **settings,
        )
    )
    (out_dir / engine_file).write_bytes(serialized)

    _write_sidecar(out_dir, bundle, name, trt_out, engine_file=engine_file)
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
