"""
Shared compile + export helpers used by every strategy.

Two valid precision paths (mutually exclusive):

  Path A — PyTorch-side cast (default)
    model = model.to(torch.float16)            # PyTorch casts weights
    ep = torch.export.export(model, ...)
    trt = torch_tensorrt.dynamo.compile(
        ep, ...,
        use_explicit_typing=True,              # respect model dtypes (incl. FP32 casts)
        use_fp32_acc=True,                     # FP32 matmul accumulation for FP16 weights
    )

  Path B — Torch-TensorRT autocast
    model stays in FP32
    trt = torch_tensorrt.dynamo.compile(
        ep, ...,
        use_explicit_typing=True,
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
    )

`enabled_precisions` is deprecated when `use_explicit_typing=True` and is
NEVER set by these helpers.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch_tensorrt

# --------------------------------------------------------------------------- #
# Precision plumbing
# --------------------------------------------------------------------------- #

PRECISION_MAP = {
    "FP16": torch.float16,
    "BF16": torch.bfloat16,
    "FP32": torch.float32,
}


def model_dtype(precision: str, autocast: bool) -> torch.dtype:
    """
    Dtype to cast the PyTorch model to before export.

    Path A (autocast=False): model is cast to the target precision.
    Path B (autocast=True): model stays in FP32; TRT compiler does the cast.
    """
    if autocast:
        return torch.float32
    return PRECISION_MAP[precision]


def compile_kwargs(precision: str, autocast: bool) -> dict:
    """
    Build the precision-related kwargs for torch_tensorrt.dynamo.compile.

    Always sets use_explicit_typing=True.  Picks Path A or Path B based on
    `autocast`.  Never sets `enabled_precisions` (deprecated under
    use_explicit_typing).
    """
    if precision not in PRECISION_MAP:
        raise ValueError(f"Unknown precision {precision!r}. Use FP16, BF16, or FP32.")

    kwargs: dict = {"use_explicit_typing": True}

    if precision == "FP32":
        # Nothing else to do; FP32 model + explicit typing.
        return kwargs

    target_dtype = PRECISION_MAP[precision]

    if autocast:
        # Path B: TRT compiler does the cast.  Model must be FP32.
        kwargs["enable_autocast"] = True
        kwargs["autocast_low_precision_type"] = target_dtype
    else:
        # Path A: model already cast in PyTorch.  Use FP32 matmul acc for FP16.
        if precision == "FP16":
            kwargs["use_fp32_acc"] = True

    return kwargs


# --------------------------------------------------------------------------- #
# Export helper (with fallback for guard violations)
# --------------------------------------------------------------------------- #


def safe_export(
    module: torch.nn.Module,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    dynamic_shapes=None,
):
    """
    torch.export.export with a two-stage fallback.

    Stage 1: strict=False (standard).
    Stage 2: If stage 1 raises a constraint / guard violation, retry with
             prefer_deferred_runtime_asserts_over_guards=True.

    WARNING: the deferred-assertions fallback converts guards to runtime
    ops that TRT may miscompile or ignore, producing wrong output for
    models with data-dependent control flow (e.g. LLaMA RoPE scaling).
    Callers should use Dim.AUTO in dynamic_shapes rather than explicit
    min/max to avoid triggering this fallback in the first place.
    """
    kwargs = kwargs or {}
    with torch.no_grad():
        try:
            return torch.export.export(
                module,
                args=args,
                kwargs=kwargs,
                dynamic_shapes=dynamic_shapes,
                strict=False,
            )
        except Exception as e:
            print(
                f"[compile] torch.export.export failed ({type(e).__name__}: "
                f"{str(e)[:120]}); retrying with deferred guards.\n"
                "[compile] WARNING: deferred-assertions export may produce wrong "
                "TRT output for RoPE / causal-SDPA models. Use Dim.AUTO in "
                "dynamic_shapes to avoid this."
            )
            return torch.export._trace._export(
                module,
                args=args,
                kwargs=kwargs,
                dynamic_shapes=dynamic_shapes,
                strict=False,
                prefer_deferred_runtime_asserts_over_guards=True,
            )


# --------------------------------------------------------------------------- #
# Compile wrapper
# --------------------------------------------------------------------------- #


def _build_trt_kwargs(
    precision: str,
    autocast: bool,
    min_block_size: int,
    debug: bool,
    offload_module_to_cpu: bool,
    cache_built_engines: bool,
    reuse_cached_engines: bool,
    engine_cache_dir: Optional[str],
    extra: Optional[dict],
    optimization_level: Optional[int] = None,
) -> dict:
    kw = compile_kwargs(precision, autocast)
    kw.update(
        {
            "device": torch.device("cuda:0"),
            "disable_tf32": True,
            "min_block_size": min_block_size,
            "debug": debug,
            "offload_module_to_cpu": offload_module_to_cpu,
            "cache_built_engines": cache_built_engines,
            "reuse_cached_engines": reuse_cached_engines,
        }
    )
    if optimization_level is not None:
        kw["optimization_level"] = optimization_level
    if engine_cache_dir is not None:
        kw["engine_cache_dir"] = engine_cache_dir
    if extra:
        kw.update(extra)
    return kw


def compile_with_trt(
    ep,
    *,
    inputs: Iterable,
    precision: str,
    autocast: bool = False,
    min_block_size: int = 1,
    optimization_level: Optional[int] = None,
    debug: bool = False,
    offload_module_to_cpu: bool = False,
    cache_built_engines: bool = True,
    reuse_cached_engines: bool = True,
    engine_cache_dir: Optional[str] = None,
    extra: Optional[dict] = None,
) -> torch.fx.GraphModule:
    """
    Thin wrapper around torch_tensorrt.dynamo.compile that always sets
    use_explicit_typing=True and applies precision plumbing consistently.

    Defaults:
      - C++ runtime (use_python_runtime=False)
      - engine caching ON
      - offload_module_to_cpu OFF (opt-in for memory-constrained models)
    """
    kw = _build_trt_kwargs(
        precision,
        autocast,
        min_block_size,
        debug,
        offload_module_to_cpu,
        cache_built_engines,
        reuse_cached_engines,
        engine_cache_dir,
        extra,
        optimization_level=optimization_level,
    )
    return torch_tensorrt.dynamo.compile(ep, inputs=list(inputs), **kw)


def maybe_save_exported_program(cfg, ep, *, log_prefix: str) -> None:
    """Save the pre-TRT ExportedProgram to cfg.save_exported_program if set."""
    path = getattr(cfg, "save_exported_program", None)
    if not path:
        return
    print(f"{log_prefix} Saving pre-TRT ExportedProgram to {path} ...")
    torch.export.save(ep, path)
    print(f"{log_prefix} ExportedProgram saved.")


def maybe_save_trt_module(
    cfg,
    module,
    *,
    arg_inputs=None,
    kwarg_inputs=None,
    example_arg_inputs=None,
    example_kwarg_inputs=None,
    log_prefix: str,
) -> None:
    """
    Save the TRT-compiled module to cfg.save_engine in cfg.engine_format
    (exported_program | torchscript | aot_inductor).

    `arg_inputs` / `kwarg_inputs` are torch_tensorrt.Input specs (used by
    the exported_program path).  `example_arg_inputs` / `example_kwarg_inputs`
    are real tensors used by the torchscript path (jit.trace needs them).
    Pass both when supporting all formats; the helper picks the right set.
    """
    path = getattr(cfg, "save_engine", None)
    if not path:
        return
    fmt = getattr(cfg, "engine_format", "exported_program")
    print(f"{log_prefix} Saving TRT module ({fmt}) to {path} ...")

    # torchscript needs real tensors for jit.trace; other formats accept
    # torch_tensorrt.Input specs.
    if fmt == "torchscript":
        use_args = example_arg_inputs if example_arg_inputs is not None else arg_inputs
        use_kwargs = (
            example_kwarg_inputs if example_kwarg_inputs is not None else kwarg_inputs
        )
    else:
        use_args = arg_inputs
        use_kwargs = kwarg_inputs

    save_kwargs = {"output_format": fmt}
    if use_kwargs:
        save_kwargs["arg_inputs"] = list(use_args or [])
        save_kwargs["kwarg_inputs"] = use_kwargs
    elif use_args:
        save_kwargs["inputs"] = list(use_args)

    torch_tensorrt.save(module, path, **save_kwargs)
    print(f"{log_prefix} TRT module saved.")


def maybe_save_trt_engine(
    cfg,
    ep,
    arg_inputs,
    *,
    log_prefix: str,
    kwarg_inputs: Optional[dict] = None,
) -> None:
    """
    If cfg.save_trt_engine is set, build and write a raw serialized TRT
    engine.  `kwarg_inputs` is required when the exported program has
    keyword args (e.g. LLMs export with position_ids as a kwarg).
    """
    path = getattr(cfg, "save_trt_engine", None)
    if not path:
        return
    print(f"{log_prefix} Serializing raw TRT engine ...")
    engine_bytes = serialize_trt_engine(
        ep,
        arg_inputs=list(arg_inputs),
        kwarg_inputs=kwarg_inputs,
        precision=cfg.precision,
        autocast=cfg.autocast,
        min_block_size=cfg.min_block_size,
        debug=cfg.debug,
        offload_module_to_cpu=cfg.offload_module_to_cpu,
    )
    with open(path, "wb") as f:
        f.write(engine_bytes)
    print(f"{log_prefix} Raw TRT engine ({len(engine_bytes)} bytes) saved to {path}")


def serialize_trt_engine(
    ep,
    *,
    arg_inputs: Iterable,
    kwarg_inputs: Optional[dict] = None,
    precision: str,
    autocast: bool = False,
    min_block_size: int = 1,
    optimization_level: Optional[int] = None,
    debug: bool = False,
    offload_module_to_cpu: bool = False,
    engine_cache_dir: Optional[str] = None,
    extra: Optional[dict] = None,
) -> bytes:
    """
    Build a single TensorRT engine for the exported program and return its
    raw serialized bytes.

    Args:
        arg_inputs: positional torch_tensorrt.Input specs matching the
            exported program's positional args.
        kwarg_inputs: dict of {name: torch_tensorrt.Input} matching the
            EP's keyword args.  Required when the EP was exported with
            kwargs (e.g. LLMs export with position_ids as a kwarg).

    Useful for deploying the engine outside the torch_tensorrt runtime
    (e.g. in TRT-LLM, the standalone TRT C++/Python API, Triton).  The
    resulting bytes are NOT a torch_tensorrt module — they are a TRT
    serialized engine produced by `IBuilder::buildSerializedNetwork`.

    Note: this is only valid when the exported program compiles into a
    single TRT engine end-to-end (no PyTorch fallback partitions).  If
    your model has unsupported ops, this call will fail; use the
    `compile_with_trt` + `torch_tensorrt.save` path instead.
    """
    kw = _build_trt_kwargs(
        precision,
        autocast,
        min_block_size,
        debug,
        offload_module_to_cpu,
        # Engine caching is irrelevant for serialization output.
        cache_built_engines=False,
        reuse_cached_engines=False,
        engine_cache_dir=engine_cache_dir,
        extra=extra,
        optimization_level=optimization_level,
    )
    return torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
        ep,
        arg_inputs=list(arg_inputs),
        kwarg_inputs=kwarg_inputs,
        **kw,
    )
