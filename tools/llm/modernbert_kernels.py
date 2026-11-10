# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Inductor's own kernels inside Torch-TensorRT ModernBERT engines.

Hybrid variants of ``agentic_encoder.py``'s ModernBERT workload (knob
``kernels.inductor``): the TensorRT config keeps the model, and one
per-layer subgraph goes to Inductor. That subgraph is the attention input's
QKV split and rotary embedding (RoPE), :func:`split_rope`: from the ``Wqkv``
projection ``qkv`` [batch, length, 3 * hidden] and the layer type's RoPE
``inv_freq`` buffer, the rotary cos/sin tables (positions ``0 .. length - 1``)
and the rotated query and key, each a contiguous [batch, length, heads,
head_dim] tensor in the model's dtype (Hugging Face's ``apply_rotary_pos_emb``
math: FP32 products, one rounding). The value stays a view of ``qkv`` in
TensorRT, and the attention reads all three as [batch, heads, length,
head_dim] views, as Hugging Face's eager forward does. :func:`install` patches
every ``ModernBertAttention`` of a loaded ``ModernBertModel`` to compute it in
one of three ways:

* ``rope-kernel``: :func:`generate_rope_kernel` compiles :func:`split_rope` at
  the workload's shapes with ``torch.compile`` (``max-autotune-no-cudagraphs``:
  Inductor's pointwise autotuning plus coordinate-descent tuning) and takes
  the single Triton kernel Inductor generates for it, with the launch
  configuration Inductor's autotuner chose. ``torch_tensorrt.region.fused_kernel``
  hosts that kernel inside the TensorRT engine as a TensorRT 11.4 QDP plugin,
  one plugin layer per transformer layer, with no engine split. The plugin
  launches a small generated Triton launcher (:class:`InductorKernel`) that
  calls Inductor's ``@triton.jit`` function unchanged: it orders the pointers
  as the plugin passes them (inputs, then outputs), bakes the element counts
  into constexprs, and restores the 16-byte pointer alignment that Inductor's
  launcher specializes on (the plugin compile path does not). The launcher
  is given Inductor's kernel name, so the PTX entry and profiles show that
  name by construction; what ties the plugin to Inductor's output is the
  sha256 of the kernel source (:attr:`InductorKernel.digest`, in the build's
  description), and torch_tensorrt, not Inductor, compiles that source.
  TensorRT copies each plugin output into the layout its attention kernel
  reads; head-major outputs (two plugins per layer) or a pre-transposed key
  (a 2-D tiled Inductor kernel) were tried and still got one copy per
  output, so the region keeps Hugging Face's token-major layout.
* ``rope-kernel-eager-numerics``: ``rope-kernel`` with the kernel generated
  under Inductor's ``emulate_precision_casts``, Inductor's option to keep
  PyTorch eager's low-precision rounding: the kernel rounds the cos/sin
  tables to FP16 as Hugging Face does, so its query and key equal Hugging
  Face's, and the engine computes what the config without the knob computes.
* ``inductor-regions``: ``torch_tensorrt.region.execute_in_torch(backend=
  "inductor")`` around the same subgraph in every layer, so it runs outside
  TensorRT, compiled by Inductor in the region compiler's settings (default
  mode, no Inductor CUDA graphs). Each region splits the engine: one region
  per layer, so two TensorRT/PyTorch boundaries per layer and one engine more
  than there are layers (22 regions, 44 boundaries and 23 engines for
  ModernBERT-base).

The same ``split_rope_{layer}`` scope can also run a kernel that PR 4914's
``torch_tensorrt.region.KernelGenerator`` writes (:data:`GENERATED`,
``agentic_encoder.py``'s ``kernels.generated="split-rope"``):
``fused_kernel(generator=...)`` around the same body, with the same inputs
(``qkv``, ``inv_freq``). The generator writes the kernel during the build,
checks it against the body and times it against TensorRT's own build of the
region; layers whose regions are identical (every layer here: ``inv_freq`` is
an input) share one kernel. Like the hosted kernel it runs inside the engine
as a QDP plugin, one per layer.

Numerics: with its default ``emulate_precision_casts=False`` Inductor elides
the FP16 round trip of the cos/sin tables that Hugging Face's rotary
embedding (and so the plain TensorRT config) performs, and computes them in
FP32 (libdevice ``cosf``/``sinf``). Query and key then differ from Hugging
Face's in the last FP16 bit for part of the elements (closer to an exact
RoPE), in ``rope-kernel`` and ``inductor-regions``. That moves ModernBERT's
few ill-conditioned tokens like any change of rounding, for better or worse
depending on the input; ``rope-kernel-eager-numerics`` avoids the change.

All need ``torch_tensorrt.compile(..., ir="dynamo", strict=True)`` (regions
are captured only by strict export) and run under CUDA graphs: TensorRT 11.4
reloads QDP plugin modules on every enqueue outside CUDA graph replay, and a
model with PyTorch regions is captured as one whole-graph CUDA graph, which
hides the per-boundary host work. Eager execution, ``torch.export`` and
``torch.compile`` run the region bodies as ordinary PyTorch code, so the
patched model computes exactly what Hugging Face computes.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import importlib.util
import math
import re
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

import torch

# Values of agentic_encoder's ``kernels.inductor`` knob.
VARIANTS = ("none", "rope-kernel", "inductor-regions", "rope-kernel-eager-numerics")
# The value of agentic_encoder's ``kernels.generated`` knob that runs
# split_rope as a kernel PR 4914's KernelGenerator writes (see install).
GENERATED = "split-rope"
# The generator's notes for that region: measured facts about its numerics.
# Part of the kernel cache key; editing the file starts a new cache.
GENERATION_NOTES = "modernbert_rope_kernel_notes.md"
# The variants that host a generated kernel, and whether Inductor generates
# it with emulate_precision_casts (eager's FP16 rounding of the cos/sin tables).
KERNEL_VARIANTS = {"rope-kernel": False, "rope-kernel-eager-numerics": True}
# torch.compile mode used to generate the hosted kernel: Inductor's
# max-autotune choices without its CUDA graphs (pointwise kernels are tuned
# by benchmarking candidate launch configs, then by coordinate descent).
GENERATION_MODE = "max-autotune-no-cudagraphs"
# The TensorRT 11.4 QDP plugin launch passes Triton's shared memory as
# dynamic shared memory without opting in to more than the 48 KB default.
SHARED_MEMORY_LIMIT = 48 * 1024
# Region inputs, in fused_kernel order (as listed in ``inputs=``).
REGION_INPUTS = ("qkv", "inv_freq")
# Generated launcher modules (importable source files, which Triton needs).
_LAUNCHER_DIR = "modernbert_kernels"
_HOSTED: dict[str, Any] = {}
_TRITON_TYPES = {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}
# Inductor's block-size constexprs of pointwise kernels.
_BLOCKS = ("YBLOCK", "XBLOCK")


# ------------------------------------------------------------------ numerics
def rotary_cos_sin(
    inv_freq: torch.Tensor, length: int, dtype: torch.dtype, scaling: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hugging Face ``ModernBertRotaryEmbedding.forward`` for positions ``0 .. length - 1``.

    [1, length, head_dim] cos and sin in ``dtype``, computed in FP32 as
    Hugging Face computes them (``position_ids`` default to an ``arange``
    shared by the batch).
    """
    position_ids = torch.arange(length, device=inv_freq.device).unsqueeze(0)
    inv = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1).float()
    positions = position_ids[:, None, :].float()
    freqs = (inv @ positions).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * scaling
    sin = emb.sin() * scaling
    return cos.to(dtype), sin.to(dtype)


def split_rope(
    qkv: torch.Tensor,
    inv_freq: torch.Tensor,
    heads: int,
    head_dim: int,
    scaling: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The hybrid subgraph: QKV split and rotary embedding of query and key.

    ``qkv`` is the [batch, length, 3 * heads * head_dim] output of ``Wqkv``.
    Returns the rotated query and key as contiguous [batch, length, heads,
    head_dim] tensors (the memory layout of Hugging Face's eager result),
    query first.
    """
    from transformers.models.modernbert.modeling_modernbert import (
        apply_rotary_pos_emb,
    )

    batch, length = qkv.shape[:2]
    cos, sin = rotary_cos_sin(inv_freq, length, qkv.dtype, scaling)
    query, key, _ = qkv.view(batch, length, 3, heads, head_dim).unbind(dim=-3)
    return apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=2)


# ------------------------------------------------- Inductor's generated kernel
@dataclasses.dataclass(frozen=True)
class InductorKernel:
    """The one Triton kernel Inductor generated for :func:`split_rope`.

    ``fn`` is Inductor's ``@triton.jit`` function as Inductor loaded it from
    its cache (``path``); ``source`` is its source. ``inputs`` and
    ``outputs`` give, for each of the kernel's pointer parameters in
    declaration order, the region input or output it receives (indices into
    :data:`REGION_INPUTS` / the region's outputs); ``numels`` are the
    element counts Inductor's wrapper passes (``xnumel``, or ``ynumel`` and
    ``xnumel`` for a 2-D tiled kernel), and ``config`` the launch
    configuration Inductor's autotuner chose (``XBLOCK`` [, ``YBLOCK``],
    ``num_warps``, ``num_stages``).
    ``dtypes`` are the Triton element types of the launcher's pointers
    (region inputs, then outputs), ``scaling`` the RoPE attention scaling
    compiled into the kernel (:func:`install` checks every layer's), and
    ``eager_numerics`` whether Inductor generated it with
    ``emulate_precision_casts``.
    """

    name: str
    fn: Any
    source: str
    path: str
    parameters: tuple[str, ...]
    inputs: tuple[tuple[str, int], ...]
    outputs: tuple[tuple[str, int], ...]
    numels: tuple[tuple[str, int], ...]
    config: Mapping[str, int]
    mode: str
    length: int
    dtypes: tuple[str, ...]
    scaling: float = 1.0
    eager_numerics: bool = False

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.source.encode()).hexdigest()[:16]

    @property
    def key(self) -> str:
        """Registry key of the hosted launcher (see :func:`hosted_kernel`)."""
        options = "-".join(f"{k}{v}" for k, v in sorted(self.config.items()))
        return f"{self.name}-{self.digest}-{options}"

    @property
    def num_outputs(self) -> int:
        return len(self.outputs)

    @property
    def grid(self) -> tuple[int, ...]:
        """Inductor's launch grid: x blocks [, y blocks] (program ids 0 [, 1])."""
        numels = dict(self.numels)
        grid = [-(-numels["xnumel"] // int(self.config["XBLOCK"]))]
        if "ynumel" in numels:
            grid.append(-(-numels["ynumel"] // int(self.config["YBLOCK"])))
        return tuple(grid)

    def launcher_source(self) -> str:
        """Source of the Triton launcher the plugin runs (see the module doc)."""
        params = [f"in{i}" for i in range(len(REGION_INPUTS))] + [
            f"out{i}" for i in range(self.num_outputs)
        ]
        bound = {name: f"in{index}" for name, index in self.inputs}
        bound.update({name: f"out{index}" for name, index in self.outputs})
        for name, _ in self.numels:
            bound[name] = name.upper()
        blocks = [name for name in self.parameters if name in _BLOCKS]
        bound.update({name: name for name in blocks})
        args = [bound[name] for name in self.parameters]
        constexprs = [name.upper() for name, _ in self.numels] + blocks
        # Pointer alignment: Inductor's launcher specializes on 16-byte aligned
        # pointers (tt.divisibility=16); the plugin compile path passes none,
        # which would leave every load and store scalar. program_id(2) is 0 in
        # the plugin's 1-D or 2-D grid; adding it keeps the hint off the
        # kernel arguments, where Triton ignores it.
        hints = "".join(
            f"    {param} = tl.multiple_of({param} + tl.program_id(2), 16)\n"
            for param in params
        )
        return (
            f'"""Plugin launcher of Inductor\'s kernel {self.name} '
            f"(source sha256 {self.digest}; generated by modernbert_kernels.py)."
            '"""\n\n'
            "import triton\n"
            "import triton.language as tl\n\n\n"
            "@triton.jit\n"
            f"def {self.name}({', '.join(params)}, "
            f"{', '.join(f'{name}: tl.constexpr' for name in constexprs)}):\n"
            f"{hints}"
            f"    _inductor_kernel({', '.join(args)})\n"
        )

    def launcher(self) -> Any:
        """The ``@triton.jit`` launcher, imported from a generated source file."""
        hosted = _HOSTED.get(self.key)
        if hosted is not None:
            return hosted
        source = self.launcher_source()
        directory = Path(tempfile.gettempdir()) / _LAUNCHER_DIR
        directory.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(source.encode()).hexdigest()[:16]
        path = directory / f"launch_{digest}.py"
        if not path.exists() or path.read_text() != source:
            path.write_text(source)
        module_name = f"_{_LAUNCHER_DIR}_launch_{digest}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Inductor's JIT function, resolved by Triton when it compiles.
        module._inductor_kernel = self.fn
        sys.modules[module_name] = module
        hosted = getattr(module, self.name)
        _HOSTED[self.key] = hosted
        return hosted

    def constexprs(self) -> dict[str, int]:
        values = {name.upper(): int(value) for name, value in self.numels}
        values.update(
            {
                name: int(self.config[name])
                for name in self.parameters
                if name in _BLOCKS
            }
        )
        return values

    def describe(self) -> dict[str, Any]:
        return {
            "inductor_kernel": self.name,
            "inductor_kernel_sha256": self.digest,
            "inductor_kernel_path": self.path,
            "inductor_mode": self.mode,
            "inductor_emulate_precision_casts": self.eager_numerics,
            "inductor_config": dict(self.config),
            "plugin_grid": list(self.grid),
        }


def hosted_kernel(key: str) -> Any:
    """``fused_kernel`` factory: the launcher registered under ``key``."""
    try:
        return _HOSTED[key]
    except KeyError:
        raise RuntimeError(
            f"No hosted Inductor kernel {key!r}; call InductorKernel.launcher() "
            "before compiling"
        ) from None


def _grid(grid: tuple[int, ...], inputs: Any, outputs: Any) -> tuple[int, ...]:
    return grid


_KERNEL_DEF = re.compile(
    r"^(\w+) = async_compile\.triton\('(\w+)', '''(.*?)''', device_str='cuda'\)",
    re.MULTILINE | re.DOTALL,
)
_KERNEL_PATH = re.compile(r"^# kernel path: (\S+)$", re.MULTILINE)


def parse_wrapper(code: str) -> dict[str, Any]:
    """Read Inductor's output code for a single-kernel graph.

    Returns the kernel ``name``, its cache ``path``, the kernel's launch
    arguments as the wrapper passes them (``run_args``), and the wrapper's
    graph inputs and returned outputs. Raises ``ValueError`` unless the
    graph is exactly one Triton kernel launch.
    """
    kernels = _KERNEL_DEF.findall(code)
    if len(kernels) != 1:
        raise ValueError(
            f"Inductor generated {len(kernels)} Triton kernels for the region; "
            "fused_kernel hosts exactly one"
        )
    variable, name, _ = kernels[0]
    if variable != name:
        raise ValueError(f"Unexpected Inductor kernel binding {variable} = {name}")
    if "extern_kernels." in code.split("def call(", 1)[-1]:
        raise ValueError("Inductor's region code calls an extern kernel")
    paths = _KERNEL_PATH.findall(code)
    call = code[code.index("def call(") :]
    inputs = re.search(r"^\s+((?:\w+, )*\w+),? = args$", call, re.MULTILINE)
    returned = re.search(r"^\s+return \(([^)]*)\)", call, re.MULTILINE)
    runs = re.findall(rf"\b{name}\.run\(([^)]*)\)", call)
    if inputs is None or returned is None or len(runs) != 1:
        raise ValueError("Could not read Inductor's wrapper call for the region")
    run_args = [
        arg.strip() for arg in runs[0].split(",") if "=" not in arg and arg.strip()
    ]
    return {
        "name": name,
        "path": paths[0] if paths else "",
        "run_args": run_args,
        "graph_inputs": [arg.strip() for arg in inputs.group(1).split(",")],
        "graph_outputs": [
            arg.strip() for arg in returned.group(1).split(",") if arg.strip()
        ],
    }


def bind_parameters(
    parameters: Sequence[str],
    constexprs: Sequence[str],
    parsed: Mapping[str, Any],
) -> tuple[
    tuple[tuple[str, int], ...],
    tuple[tuple[str, int], ...],
    tuple[tuple[str, int], ...],
]:
    """Map a kernel's runtime parameters to region inputs, outputs and element counts.

    ``parameters`` are the kernel's declared parameters, ``constexprs`` the
    constexpr ones, ``parsed`` :func:`parse_wrapper`'s result. The kernel
    must be a pointwise kernel, 1-D (``..., xnumel, XBLOCK``) or 2-D tiled
    (``..., ynumel, xnumel, YBLOCK, XBLOCK``); its pointer parameters must
    receive graph inputs or returned buffers, and its element counts be
    integer literals. Returns (inputs, outputs, numels).
    """
    runtime = [name for name in parameters if name not in constexprs]
    numel_names = [name for name in runtime if name in ("ynumel", "xnumel")]
    if (
        tuple(constexprs) not in (("XBLOCK",), ("YBLOCK", "XBLOCK"))
        or runtime[-len(numel_names) :]
        != [block[0].lower() + "numel" for block in constexprs]
        or not numel_names
    ):
        raise ValueError(
            "Expected a pointwise kernel (..., xnumel, XBLOCK) or "
            f"(..., ynumel, xnumel, YBLOCK, XBLOCK); got {list(parameters)}"
        )
    run_args = parsed["run_args"]
    if len(run_args) != len(runtime):
        raise ValueError(
            f"Inductor launches {parsed['name']} with {len(run_args)} arguments "
            f"for {len(runtime)} runtime parameters"
        )
    graph_inputs = list(parsed["graph_inputs"])
    graph_outputs = list(parsed["graph_outputs"])
    pointers = len(runtime) - len(numel_names)
    inputs, outputs = [], []
    for name, arg in zip(runtime[:pointers], run_args[:pointers]):
        if arg in graph_inputs:
            inputs.append((name, graph_inputs.index(arg)))
        elif arg in graph_outputs:
            outputs.append((name, graph_outputs.index(arg)))
        else:
            raise ValueError(
                f"Kernel parameter {name} receives {arg}, neither a region input "
                "nor a region output: the region is more than one kernel"
            )
    if sorted(index for _, index in inputs) != list(range(len(graph_inputs))) or sorted(
        index for _, index in outputs
    ) != list(range(len(graph_outputs))):
        raise ValueError(
            "The kernel does not read every region input and write every output once"
        )
    if any(name.startswith("in_out") for name, _ in inputs + outputs):
        raise ValueError("The kernel updates a buffer in place")
    numels = []
    for name, arg in zip(numel_names, run_args[pointers:]):
        try:
            numels.append((name, int(arg)))
        except ValueError:
            raise ValueError(f"{name} is not static: {arg!r}") from None
    return tuple(inputs), tuple(outputs), tuple(numels)


def _autotuner(name: str) -> Any:
    """Inductor's ``CachingAutotuner`` for kernel ``name`` (after it ran)."""
    from torch._inductor.codecache import PyCodeCache

    for module in reversed(list(PyCodeCache.modules)):
        tuner = getattr(module, name, None)
        if tuner is not None and getattr(tuner, "launchers", None):
            return tuner
    raise RuntimeError(f"Inductor's autotuner for {name} was not found")


def generate_rope_kernel(
    length: int,
    heads: int,
    head_dim: int,
    *,
    scaling: float = 1.0,
    dtype: torch.dtype = torch.float16,
    inv_freq_dtype: torch.dtype = torch.float32,
    device: Any = "cuda",
    mode: str = GENERATION_MODE,
    eager_numerics: bool = False,
    check: bool = True,
) -> InductorKernel:
    """Compile :func:`split_rope` with Inductor and return its one Triton kernel.

    The kernel's code and launch configuration are Inductor's (the FX graph
    cache is bypassed so the output code is generated; the autotuner's
    local cache applies). ``eager_numerics`` sets Inductor's
    ``emulate_precision_casts``, so the kernel keeps eager's FP16 rounding of
    the cos/sin tables. ``check`` launches the kernel through the hosted
    launcher on random inputs and compares it with :func:`split_rope`.
    """
    import torch._inductor.config as inductor_config
    from torch._inductor.utils import run_and_get_code

    hidden = heads * head_dim
    generator = torch.Generator(device=device).manual_seed(0)
    qkv = torch.randn(
        1, length, 3 * hidden, device=device, dtype=dtype, generator=generator
    )
    inv_freq = (
        1.0
        / (
            10000.0
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
                / head_dim
            )
        )
    ).to(inv_freq_dtype)

    def compiled_region(qkv: torch.Tensor, inv_freq: torch.Tensor) -> Any:
        return split_rope(qkv, inv_freq, heads, head_dim, scaling)

    with (
        inductor_config.patch(
            fx_graph_cache=False, emulate_precision_casts=bool(eager_numerics)
        ),
        torch.no_grad(),
    ):
        compiled = torch.compile(
            compiled_region, mode=mode, fullgraph=True, dynamic=False
        )
        _, codes = run_and_get_code(compiled, qkv, inv_freq)
    if len(codes) != 1:
        raise ValueError(f"Inductor produced {len(codes)} graphs for the region")
    parsed = parse_wrapper(codes[0])
    if parsed["graph_inputs"] != ["arg0_1", "arg1_1"]:
        raise ValueError(f"Unexpected region inputs {parsed['graph_inputs']}")
    tuner = _autotuner(parsed["name"])
    fn = tuner.fn
    constexprs = [p.name for p in fn.params if p.is_constexpr]
    inputs, outputs, numels = bind_parameters(fn.arg_names, constexprs, parsed)
    launch = tuner.launchers[0].config
    config = {
        **{k: int(v) for k, v in launch.kwargs.items()},
        "num_warps": int(launch.num_warps),
        "num_stages": int(launch.num_stages),
    }
    if set(config) != {*constexprs, "num_warps", "num_stages"}:
        raise ValueError(f"Unexpected launch configuration {config}")
    kernel = InductorKernel(
        name=parsed["name"],
        fn=fn,
        source=fn.src,
        path=parsed["path"],
        parameters=tuple(fn.arg_names),
        inputs=inputs,
        outputs=outputs,
        numels=numels,
        config=config,
        mode=mode,
        length=length,
        dtypes=(_TRITON_TYPES[dtype], _TRITON_TYPES[inv_freq_dtype])
        + (_TRITON_TYPES[dtype],) * len(outputs),
        scaling=float(scaling),
        eager_numerics=bool(eager_numerics),
    )
    elements = math.prod(value for _, value in numels)
    if elements % length:
        raise ValueError(f"{kernel.name}: {dict(numels)} elements for {length} tokens")
    if check:
        check_kernel(kernel, qkv, inv_freq, heads, head_dim, scaling)
    return kernel


def check_kernel(
    kernel: InductorKernel,
    qkv: torch.Tensor,
    inv_freq: torch.Tensor,
    heads: int,
    head_dim: int,
    scaling: float = 1.0,
) -> None:
    """Launch the hosted launcher once and compare it with :func:`split_rope`.

    Without ``eager_numerics`` Inductor computes the cos/sin tables in FP32
    (it does not round them to FP16 as eager does), so the comparison allows
    FP16-rounding differences; a wrong pointer binding fails it by far.
    """
    expected = split_rope(qkv, inv_freq, heads, head_dim, scaling)
    actual = [torch.empty_like(tensor) for tensor in expected]
    kernel.launcher()[kernel.grid](
        qkv,
        inv_freq,
        *actual,
        **kernel.constexprs(),
        num_warps=kernel.config["num_warps"],
        num_stages=kernel.config["num_stages"],
    )
    for index, (a, e) in enumerate(zip(actual, expected)):
        torch.testing.assert_close(
            a.float(),
            e.float(),
            rtol=1e-2,
            atol=1e-2,
            msg=lambda m, index=index: f"{kernel.name} output {index}: {m}",
        )


def kernel_resources(kernel: InductorKernel, target: Any = None) -> dict[str, Any]:
    """Compile the launcher as the plugin does and report what it needs.

    Returns Triton's ``shared`` memory and scratch sizes, the PTX entry name
    and how many of the PTX's global loads and stores are vectorized.
    """
    import triton

    if target is None:
        target = triton.runtime.driver.active.get_current_target()
    names = [f"in{i}" for i in range(len(REGION_INPUTS))] + [
        f"out{i}" for i in range(kernel.num_outputs)
    ]
    signature = {name: f"*{dtype}" for name, dtype in zip(names, kernel.dtypes)}
    compiled = triton.compile(
        triton.compiler.ASTSource(
            fn=kernel.launcher(), signature=signature, constexprs=kernel.constexprs()
        ),
        target=target,
        options={
            "arch": f"sm{target.arch}",
            "num_warps": kernel.config["num_warps"],
            "num_stages": kernel.config["num_stages"],
        },
    )
    metadata = compiled.metadata
    return {
        "entry": metadata.name,
        "shared": int(metadata.shared),
        "global_scratch_size": int(getattr(metadata, "global_scratch_size", 0) or 0),
        "profile_scratch_size": int(getattr(metadata, "profile_scratch_size", 0) or 0),
        "global_access_bits": access_widths(compiled.asm["ptx"]),
    }


def access_widths(ptx: str) -> dict[str, dict[str, int]]:
    """Global load and store instructions of ``ptx`` by width in bits.

    ``{"ld": {"32": 10, "64": 1}, "st": {"32": 2}}``: a ``ld.global.v2.b32``
    moves 64 bits. Scalar FP16 accesses show up as 16-bit ones.
    """
    widths: dict[str, dict[str, int]] = {"ld": {}, "st": {}}
    for op, suffix in re.findall(r"\b(ld|st)\.global((?:\.[\w:]+)+)", ptx):
        bits = re.search(r"\.[bfsu](\d+)$", suffix)
        if bits is None:
            continue
        vector = re.search(r"\.v(\d+)\.", suffix + ".")
        width = str(int(bits.group(1)) * (int(vector.group(1)) if vector else 1))
        widths[op][width] = widths[op].get(width, 0) + 1
    return widths


def check_resources(kernel: InductorKernel, **kwargs: Any) -> dict[str, Any]:
    """:func:`kernel_resources`, refusing what the TensorRT plugin cannot launch."""
    resources = kernel_resources(kernel, **kwargs)
    if resources["shared"] > SHARED_MEMORY_LIMIT:
        raise ValueError(
            f"{kernel.name} needs {resources['shared']} bytes of shared memory; the "
            f"TensorRT 11.4 plugin launch allows {SHARED_MEMORY_LIMIT}"
        )
    if resources["global_scratch_size"] or resources["profile_scratch_size"]:
        raise ValueError(
            f"{kernel.name} needs Triton scratch memory; the TensorRT plugin "
            "launch passes null scratch pointers"
        )
    return resources


# ------------------------------------------------------------ model patching
@dataclasses.dataclass(frozen=True)
class _Hybrid:
    """How a patched ``ModernBertAttention`` computes :func:`split_rope`.

    Plain constants only: the forward reads them while strict export traces
    it, so the hosted kernel's ``fused_kernel`` options are computed in
    :func:`install`.
    """

    variant: str
    heads: int
    scaling: float
    kernel_key: str = ""
    constexprs: tuple[tuple[str, int], ...] = ()
    grid: tuple[int, ...] = ()
    num_warps: int = 0
    num_stages: int = 0
    # The KernelGenerator of the GENERATED variant (created before tracing).
    generator: Any = dataclasses.field(default=None, compare=False)

    def scope(self, layer: int, qkv: torch.Tensor, inv_freq: torch.Tensor) -> Any:
        from torch_tensorrt import region

        if self.variant == GENERATED:
            # The body stays the eager reference; the generator writes,
            # checks and caches the kernel that replaces it at compile time.
            return region.fused_kernel(
                generator=self.generator,
                inputs=[qkv, inv_freq],
                name=f"split_rope_{layer}",
            )
        if self.variant in KERNEL_VARIANTS:
            return region.fused_kernel(
                kernel=functools.partial(hosted_kernel, self.kernel_key),
                inputs=[qkv, inv_freq],
                name=f"split_rope_{layer}",
                constexprs=dict(self.constexprs),
                grid=functools.partial(_grid, self.grid),
                num_warps=self.num_warps,
                num_stages=self.num_stages,
            )
        if self.variant == "inductor-regions":
            return region.execute_in_torch(
                name=f"split_rope_{layer}", backend="inductor"
            )
        return contextlib.nullcontext()


def _attention_forward(
    self: Any,
    hidden_states: torch.Tensor,
    position_embeddings: Any = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, Any]:
    """``ModernBertAttention.forward`` with :func:`split_rope` as a region.

    Hugging Face's forward except that query and key come from the region
    (the layer's own RoPE buffer replaces ``position_embeddings``); the
    value and the attention call are Hugging Face's.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.modernbert.modeling_modernbert import (
        eager_attention_forward,
    )

    hybrid: _Hybrid = self._modernbert_hybrid
    input_shape = hidden_states.shape[:-1]
    qkv = self.Wqkv(hidden_states)
    inv_freq = self.hybrid_inv_freq
    with hybrid.scope(self.layer_idx, qkv, inv_freq):
        query, key = split_rope(
            qkv, inv_freq, hybrid.heads, self.head_dim, hybrid.scaling
        )
    value = qkv.view(*input_shape, 3, -1, self.head_dim).unbind(dim=-3)[2]
    query, key, value = (
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
    )
    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self,
        query,
        key,
        value,
        attention_mask,
        dropout=self.attention_dropout if self.training else 0.0,
        scaling=self.head_dim**-0.5,
        sliding_window=self.sliding_window,
        deterministic=self.deterministic_flash_attn,
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.out_drop(self.Wo(attn_output))
    return attn_output, attn_weights


def install(
    model: torch.nn.Module,
    variant: str,
    kernel: Optional[InductorKernel] = None,
    *,
    generator: Any = None,
) -> int:
    """Patch every ``ModernBertAttention`` of ``model`` (a ``ModernBertModel``).

    ``variant`` is one of :data:`VARIANTS` or :data:`GENERATED`; the
    :data:`KERNEL_VARIANTS` need the ``kernel`` from
    :func:`generate_rope_kernel` (its launcher is registered here), generated
    with the variant's ``eager_numerics``, and every layer's RoPE must match
    it: the same attention scaling and ``inv_freq`` dtype. :data:`GENERATED`
    needs a ``torch_tensorrt.region.KernelGenerator`` (``generator``), which
    writes the kernel of every layer's region during compilation. Each
    attention gets its layer type's RoPE ``inv_freq`` as a non-persistent
    buffer ``hybrid_inv_freq``. Every layer is checked before any is patched.
    ``"none"`` changes nothing. Returns the number of patched layers.
    """
    if variant not in VARIANTS and variant != GENERATED:
        raise ValueError(
            f"Unknown variant {variant!r}; choose one of {VARIANTS + (GENERATED,)}"
        )
    if variant == "none":
        return 0
    if (variant == GENERATED) != (generator is not None):
        raise ValueError(
            f"{variant} {'needs a generator' if generator is None else 'takes no generator'}; "
            f"only {GENERATED!r} generates its kernel"
        )
    if (variant in KERNEL_VARIANTS) != (kernel is not None):
        raise ValueError(
            f"{variant} {'needs a kernel' if kernel is None else 'takes no kernel'}; "
            f"only {list(KERNEL_VARIANTS)} host one"
        )
    if kernel is not None and kernel.eager_numerics != KERNEL_VARIANTS[variant]:
        raise ValueError(
            f"{variant} needs a kernel generated with eager_numerics="
            f"{KERNEL_VARIANTS[variant]}"
        )
    config = model.config
    settings = layer_rope(model)
    if kernel is not None:
        for layer_type, (scaling, inv_freq) in sorted(settings.items()):
            if scaling != kernel.scaling:
                raise ValueError(
                    f"{layer_type} layers scale RoPE by {scaling}; the hosted kernel "
                    f"was generated for {kernel.scaling}"
                )
            if _TRITON_TYPES.get(inv_freq.dtype) != kernel.dtypes[1]:
                raise ValueError(
                    f"{layer_type} layers have {inv_freq.dtype} inv_freq; the hosted "
                    f"kernel reads {kernel.dtypes[1]}"
                )
    plugin: dict[str, Any] = {}
    if kernel is not None:
        kernel.launcher()
        plugin = dict(
            kernel_key=kernel.key,
            constexprs=tuple(sorted(kernel.constexprs().items())),
            grid=kernel.grid,
            num_warps=int(kernel.config["num_warps"]),
            num_stages=int(kernel.config["num_stages"]),
        )
    patched = 0
    for layer in model.layers:
        scaling, inv_freq = settings[layer.attention_type]
        attention = layer.attn
        attention.register_buffer(
            "hybrid_inv_freq", inv_freq.detach().clone(), persistent=False
        )
        hybrid = _Hybrid(
            variant=variant,
            heads=int(config.num_attention_heads),
            scaling=scaling,
            generator=generator,
            **plugin,
        )
        # A plain attribute: it is neither a module nor a tensor.
        object.__setattr__(attention, "_modernbert_hybrid", hybrid)
        attention.forward = types.MethodType(_attention_forward, attention)
        patched += 1
    return patched


def layer_rope(model: torch.nn.Module) -> dict[str, tuple[float, torch.Tensor]]:
    """Each layer type's RoPE attention scaling and ``inv_freq`` buffer.

    Raises ``ValueError`` for a layer type whose rotary embedding is not
    the default one, which is all :func:`split_rope` implements.
    """
    rotary = model.rotary_emb
    settings = {}
    for layer_type in sorted({layer.attention_type for layer in model.layers}):
        rope_type = rotary.rope_type.get(layer_type)
        if rope_type != "default":
            raise ValueError(
                f"{layer_type} layers use RoPE type {rope_type!r}; the hybrid "
                "subgraph implements the default rotary embedding"
            )
        settings[layer_type] = (
            float(getattr(rotary, f"{layer_type}_attention_scaling")),
            getattr(rotary, f"{layer_type}_inv_freq"),
        )
    return settings


@contextlib.contextmanager
def no_hosted_kernels() -> Iterator[None]:
    """Clear the launcher registry afterwards (tests)."""
    try:
        yield
    finally:
        _HOSTED.clear()
