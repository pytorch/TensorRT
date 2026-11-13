from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch_tensorrt
from torch_tensorrt.hf.exporters.ops import _as_tuple, record_engine
from torch_tensorrt.hf.exporters.spec import ComponentBundle

DEFAULT_TRT_SETTINGS: dict[str, Any] = {
    "min_block_size": 1,
    "require_full_compilation": True,
    "immutable_weights": True,
    "disable_tf32": True,
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
    dryrun: bool = False,
    trt_settings: dict[str, Any] | None = None,
) -> tuple[str, tuple[torch.Tensor, ...]]:
    """Export one component, compile it, write ``engine_dir/<name>/``.

    Returns ``(engine_dir, example_outputs)`` from a patched eager run so the
    exporter can chain components without a second unpatched forward.

    Family setattr is owned by ``EdgeSpec.apply_patches``, not this helper.
    ``dryrun`` records the patched eager module for ``execute_engine``.
    """
    from torch_tensorrt.hf.exporters.plugin.attn_patches import (
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

    patched = bundle.patch_fn(module) if bundle.patch_fn is not None else None
    try:
        with torch.no_grad():
            example = module(*execute_args)
        outputs = _as_tuple(example)
        record_engine(
            engine_path,
            component=name,
            input_names=bundle.input_names,
            outputs=outputs,
            module=module,
        )
        if dryrun:
            _write_sidecar(out_dir, bundle, name, outputs, dryrun=True)
            return engine_path, outputs

        exported = torch.export.export(module, args=trace_args, strict=False)
        settings = {
            k: v
            for k, v in {
                **DEFAULT_TRT_SETTINGS,
                **(trt_settings or {}),
                **bundle.trt_settings,
            }.items()
            if k in _TRT_COMPILE_KEYS
        }

        compiled = torch_tensorrt.dynamo.compile(
            exported,
            arg_inputs=trace_args,
            **settings,
        )
        record_engine(
            engine_path,
            component=name,
            input_names=bundle.input_names,
            outputs=outputs,
            module=compiled,
        )
        engine_file = bundle.engine_file

        serialized = (
            torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
                exported,
                arg_inputs=trace_args,
                **settings,
            )
        )
        (out_dir / engine_file).write_bytes(serialized)

        _write_sidecar(out_dir, bundle, name, outputs, engine_file=engine_file)
        return engine_path, outputs
    finally:
        if not dryrun and patched is not None:
            from torch_tensorrt.hf.exporters.plugin.plugin_utils import (
                restore_attention,
            )

            restore_attention(patched)  # type: ignore[no-untyped-call]


def _write_sidecar(
    out_dir: Path,
    bundle: ComponentBundle,
    name: str,
    outputs: tuple[torch.Tensor, ...],
    *,
    engine_file: str | None = None,
    dryrun: bool = False,
) -> None:
    config = {
        "model_type": bundle.model_type,
        "component": name,
        "engine_file": engine_file or bundle.engine_file,
        "input_names": list(bundle.input_names),
        "output_names": list(bundle.output_names),
        "dryrun": dryrun,
        "outputs": [{"shape": list(t.shape), "dtype": str(t.dtype)} for t in outputs],
    }
    config.update(bundle.extra_config)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
