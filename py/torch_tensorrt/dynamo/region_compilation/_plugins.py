# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Derive a region's Torch contract and register one custom-kernel plugin."""

from __future__ import annotations

import copy
import inspect
import logging
import uuid
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Sequence

import torch
from torch_tensorrt.region._errors import RegionError

from ._validation import validate_output_aliases

if TYPE_CHECKING:
    from torch_tensorrt.kernels import KernelSpec

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TensorBoundary:
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device

    @classmethod
    def from_value(cls, value: Any, *, where: str) -> _TensorBoundary:
        if not isinstance(value, torch.Tensor):
            raise RegionError(f"{where} requires Tensor metadata")
        if any(not isinstance(dim, int) for dim in value.shape):
            raise RegionError(f"{where} requires static tensor shapes")
        if value.device.type != "cuda":
            raise RegionError(f"{where} requires CUDA tensors")
        if value.layout != torch.strided or not value.is_contiguous():
            raise RegionError(f"{where} requires contiguous strided tensors")
        if value.is_conj() or value.is_neg():
            raise RegionError(f"{where} requires resolved conjugate and negative views")
        return cls(tuple(value.shape), value.dtype, value.device)

    def matches(self, value: Any) -> bool:
        try:
            return self == self.from_value(value, where="Plugin region boundary")
        except RegionError:
            return False


def _matches_boundaries(
    specs: Sequence[_TensorBoundary], values: Sequence[Any]
) -> bool:
    return len(specs) == len(values) and all(
        spec.matches(value) for spec, value in zip(specs, values)
    )


def _boundaries(
    child: torch.fx.GraphModule, region_id: str
) -> tuple[list[_TensorBoundary], list[_TensorBoundary]]:
    inputs = [node for node in child.graph.nodes if node.op == "placeholder"]
    output = next(node for node in child.graph.nodes if node.op == "output")
    returned = output.args[0]
    outputs = list(returned) if isinstance(returned, (tuple, list)) else [returned]
    if not inputs or not outputs:
        raise RegionError(
            f"Plugin region {region_id!r} requires at least one Tensor input and output"
        )
    if any(not isinstance(node, torch.fx.Node) for node in outputs):
        raise RegionError(
            f"Plugin region {region_id!r} requires a flat tuple of Tensor outputs"
        )
    input_values = [node.meta.get("val") for node in inputs]
    output_values = [node.meta.get("val") for node in outputs]
    input_specs = [
        _TensorBoundary.from_value(
            value, where=f"Plugin region {region_id!r} input {i}"
        )
        for i, value in enumerate(input_values)
    ]
    output_specs = [
        _TensorBoundary.from_value(
            value, where=f"Plugin region {region_id!r} output {i}"
        )
        for i, value in enumerate(output_values)
    ]
    if len({spec.device for spec in input_specs + output_specs}) != 1:
        raise RegionError(
            f"Plugin region {region_id!r} cannot span multiple CUDA devices"
        )
    validate_output_aliases(input_values, output_values, region_id)
    return input_specs, output_specs


def _make_callbacks(
    child: torch.fx.GraphModule,
    input_specs: list[_TensorBoundary],
    output_specs: list[_TensorBoundary],
    region_id: str,
) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., bool]]:
    # Later compiler rewrites must not change the eagerly executable reference.
    reference = copy.deepcopy(child)

    def pack(outputs: list[torch.Tensor]) -> Any:
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def meta_fn(*inputs: torch.Tensor) -> Any:
        # QDP probes symbolic inputs; return this specialization's static outputs.
        return pack(
            [
                torch.empty(spec.shape, dtype=spec.dtype, device=inputs[0].device)
                for spec in output_specs
            ]
        )

    def eager_fn(*inputs: torch.Tensor) -> Any:
        if not _matches_boundaries(input_specs, inputs):
            raise RegionError(
                f"Plugin region {region_id!r} inputs do not match its captured "
                "static shapes, dtypes, CUDA device, and contiguous layout"
            )
        result = reference(*inputs)
        outputs = list(result) if isinstance(result, (tuple, list)) else [result]
        if not _matches_boundaries(output_specs, outputs):
            raise RegionError(
                f"Plugin region {region_id!r} reference outputs no longer match "
                "the captured tensor metadata"
            )
        return pack(outputs)

    def capability_validator(node: torch.fx.Node, settings: Any = None) -> bool:
        inputs = [
            arg.meta.get("val") if isinstance(arg, torch.fx.Node) else arg
            for arg in node.args
        ]
        produced = node.meta.get("val")
        outputs = list(produced) if isinstance(produced, (tuple, list)) else [produced]
        if node.kwargs or any(
            not _matches_boundaries(specs, values)
            for specs, values in ((input_specs, inputs), (output_specs, outputs))
        ):
            _LOGGER.warning(
                "Plugin region %r requires its captured static shapes, dtypes, "
                "CUDA device, and contiguous layout",
                region_id,
            )
            return False
        target_device = getattr(getattr(settings, "device", None), "gpu_id", None)
        if target_device is not None and target_device >= 0:
            if input_specs[0].device.index != target_device:
                _LOGGER.warning(
                    "Plugin region %r targets a different CUDA device", region_id
                )
                return False
        if getattr(settings, "truncate_double", False) and any(
            spec.dtype == torch.float64 for spec in input_specs + output_specs
        ):
            return False
        return True

    return meta_fn, eager_fn, capability_validator


def _triton_signature(
    registration: partial[None],
    input_specs: list[_TensorBoundary],
    output_specs: list[_TensorBoundary],
    region_id: str,
) -> dict[str, str]:
    from torch_tensorrt.kernels import _triton

    options = registration.keywords
    constexprs = options["constexprs"]
    if options.get("signature") is not None:
        signature = dict(options["signature"])
    else:
        runtime_names = [
            name for name in options["kernel"].arg_names if name not in constexprs
        ]
        num_extras = len(runtime_names) - len(input_specs) - len(output_specs)
        if num_extras < 0:
            raise RegionError(
                f"Plugin region {region_id!r} has more tensor boundaries than the "
                "Triton kernel has runtime parameters"
            )
        # Encode boundary dtypes without depending on the analyzer's private tables.
        # analyze_signature below remains authoritative for the supported QDP ABI.
        dtype_names = {
            torch.float16: "fp16",
            torch.bfloat16: "bf16",
            torch.float32: "fp32",
            torch.bool: "i1",
            torch.int8: "i8",
            torch.int32: "i32",
            torch.int64: "i64",
            torch.uint8: "u8",
            torch.float8_e4m3fn: "fp8e4nv",
        }
        try:
            input_types = [f"*{dtype_names[item.dtype]}" for item in input_specs]
            output_types = [f"*{dtype_names[item.dtype]}" for item in output_specs]
        except KeyError as error:
            raise RegionError(
                f"Plugin region {region_id!r} has a dtype unsupported by Triton QDP: "
                f"{error.args[0]}"
            ) from error
        signature = dict(
            zip(runtime_names, input_types + ["i32"] * num_extras + output_types)
        )
    # Launch validation belongs to triton_op; only compare captured dtypes here.
    layout = _triton.analyze_signature(signature, (len(input_specs), len(output_specs)))
    for kind, params, boundaries in (
        ("input", layout.inputs, input_specs),
        ("output", layout.outputs, output_specs),
    ):
        for i, (param, boundary) in enumerate(zip(params, boundaries)):
            if param.dtype != boundary.dtype:
                raise RegionError(
                    f"Plugin region {region_id!r} {kind} {i} is {boundary.dtype}, "
                    f"but signature parameter {param.name!r} declares {param.dtype}"
                )
    return signature


def register_region_plugin(
    child: torch.fx.GraphModule,
    registration: KernelSpec | partial[None],
    *,
    region_id: str,
) -> torch._ops.OpOverload:
    """Register a static specialization; the caller must enforce TRT placement."""
    from torch_tensorrt import kernels
    from torch_tensorrt.kernels._dsl import InputDecl, KernelSpec

    input_specs, output_specs = _boundaries(child, region_id)
    meta_fn, eager_fn, validator = _make_callbacks(
        child, input_specs, output_specs, region_id
    )
    arguments = ", ".join(f"Tensor arg{i}" for i in range(len(input_specs)))
    returns = (
        "Tensor"
        if len(output_specs) == 1
        else "(" + ", ".join("Tensor" for _ in output_specs) + ")"
    )
    schema = f"({arguments}) -> {returns}"
    namespace = "torch_tensorrt_region_fused"
    name = f"region_{uuid.uuid4().hex}"
    op_name = f"{namespace}::{name}"
    common: dict[str, Any] = dict(
        meta_fn=meta_fn,
        eager_fn=eager_fn,
        schema=schema,
        supports_dynamic_shapes=False,
        capability_validator=validator,
    )
    with torch.cuda.device(input_specs[0].device):
        func = registration.func if isinstance(registration, partial) else None
        if isinstance(registration, KernelSpec):
            declared_inputs = registration.inputs
            if declared_inputs is None:
                declared_inputs = [
                    InputDecl(f"arg{i}", boundary.dtype)
                    for i, boundary in enumerate(input_specs)
                ]
            if len(declared_inputs) != len(input_specs) or any(
                not isinstance(decl, InputDecl)
                or (decl.dtype is not None and decl.dtype != boundary.dtype)
                for decl, boundary in zip(declared_inputs, input_specs)
            ):
                raise RegionError(
                    f"Plugin region {region_id!r} KernelSpec.inputs must describe "
                    "the region's Tensor inputs in the order inputs= lists them"
                )
            if registration.outputs is not None and len(registration.outputs) != len(
                output_specs
            ):
                raise RegionError(
                    f"Plugin region {region_id!r} KernelSpec.outputs has the wrong arity"
                )
            if registration.geometry is None:
                raise RegionError(
                    f"Plugin region {region_id!r} requires KernelSpec.geometry"
                )
            from torch_tensorrt.kernels import _common, _derive

            cuda_spec = replace(registration, inputs=declared_inputs)
            # Guard the final engine pointer dtypes, as the Triton frontend does.
            kernels.cuda_kernel_op(
                op_name,
                cuda_spec,
                aot_fn=_common.check_aot_dtypes(
                    op_name,
                    _derive._make_aot_fn(cuda_spec),
                    [boundary.dtype for boundary in input_specs],
                    [boundary.dtype for boundary in output_specs],
                ),
                **common,
            )
        elif func is kernels.cutile_op:
            # Without an explicit signature, use the generated-kernel array names.
            signature = registration.keywords.get("signature") or {
                **{f"arg{i}": spec.dtype for i, spec in enumerate(input_specs)},
                **{f"out{i}": spec.dtype for i, spec in enumerate(output_specs)},
            }
            # cuTile matches the schema's and callback's input names with the
            # kernel's input arrays, so both take their names from the signature.
            names = list(signature)[: len(input_specs)]
            meta_fn.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
                [
                    inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                    for name in names
                ]
            )
            arguments = ", ".join(f"Tensor {name}" for name in names)
            registration(
                op_name,
                signature=signature,
                **{**common, "schema": f"({arguments}) -> {returns}"},
            )
        elif func is kernels.ptx_op:
            from torch_tensorrt.kernels._common import check_aot_dtypes

            # Generated CUDA already validated and compiled this pointer ABI.
            # Keep the engine binding guard when using the precompiled funnel.
            registration(
                op_name,
                aot_fn=check_aot_dtypes(
                    op_name,
                    registration.keywords["aot_fn"],
                    [boundary.dtype for boundary in input_specs],
                    [boundary.dtype for boundary in output_specs],
                ),
                **common,
            )
        elif func is kernels.triton_op:
            # fused_kernel defaults grid to None because a CUDA KernelSpec omits it.
            if not callable(registration.keywords.get("grid")):
                raise RegionError(
                    f"Plugin region {region_id!r} requires a grid callback for "
                    "its Triton kernel"
                )
            signature = _triton_signature(
                registration, input_specs, output_specs, region_id
            )
            registration(
                op_name,
                signature=signature,
                **common,
            )
        else:
            raise RegionError(
                f"Plugin region {region_id!r} registration must be a KernelSpec or "
                "a functools.partial of kernels.triton_op, kernels.cutile_op, or "
                f"kernels.ptx_op; got {registration!r:.200}"
            )
    return getattr(getattr(torch.ops, namespace), name).default
