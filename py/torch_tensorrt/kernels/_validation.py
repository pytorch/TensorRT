"""Fail-fast validation for :class:`KernelSpec`.

Runs before any compilation or registration so authoring mistakes surface
with actionable error messages instead of late KeyErrors or silent
miscomputation at launch time.
"""

from __future__ import annotations

from torch_tensorrt.kernels._derive import _resolve_input_ref, _tensor_input_decls
from torch_tensorrt.kernels._dsl import (
    Custom,
    Elementwise,
    KernelSpec,
    ReduceDims,
    Reduction,
    SameAs,
    ScalarInput,
)


def _validate_spec(
    spec: KernelSpec,
    *,
    has_meta_fn: bool = False,
    has_eager_fn: bool = False,
    has_aot_fn: bool = False,
) -> None:
    """Validate a :class:`KernelSpec` before any compilation or registration.

    Catches the common authoring mistakes that would otherwise surface as
    late KeyErrors or silent miscomputation at launch time.

    The ``has_*`` flags describe which auto-derivations are bypassed by
    user-supplied overrides. A DSL field is only required when its
    derivation isn't overridden — e.g. ``outputs`` is required only when
    ``meta_fn`` is None.
    """
    needs_outputs = not has_meta_fn
    needs_geometry = not (has_eager_fn and has_aot_fn)
    needs_inputs = needs_outputs or needs_geometry

    if needs_inputs and not spec.inputs:
        raise ValueError(
            "KernelSpec.inputs is required when meta_fn / eager_fn / aot_fn are "
            "not all provided"
        )
    if needs_outputs and not spec.outputs:
        raise ValueError("KernelSpec.outputs is required when meta_fn is not provided")
    if needs_geometry and spec.geometry is None:
        raise ValueError(
            "KernelSpec.geometry is required when eager_fn or aot_fn is not provided"
        )

    if spec.inputs:
        tensor_decls = _tensor_input_decls(spec.inputs)
        if needs_inputs and not tensor_decls:
            raise ValueError(
                "KernelSpec.inputs must contain at least one tensor InputDecl; "
                "ScalarInput alone is insufficient (shape inference needs a tensor)."
            )

        input_names = {d.name for d in spec.inputs}
        if len(input_names) != len(spec.inputs):
            raise ValueError("KernelSpec.inputs contains duplicate names")

        # Scalar inputs must have a supported py_type.
        for d in spec.inputs:
            if isinstance(d, ScalarInput) and d.py_type not in (float, int, bool):
                raise ValueError(
                    f"ScalarInput {d.name!r}: py_type must be float, int, or bool, "
                    f"got {d.py_type!r}"
                )

        tensor_input_names = {d.name for d in tensor_decls}
    else:
        tensor_decls = []
        tensor_input_names = set()

    if spec.outputs:
        output_names = {d.name for d in spec.outputs}
        if len(output_names) != len(spec.outputs):
            raise ValueError("KernelSpec.outputs contains duplicate names")

        # SameAs/ReduceDims reference an entry in the *tensor* input list (by
        # index or by name); scalars are not addressable from shape decls.
        for decl in spec.outputs:
            rel = decl.shape
            if isinstance(rel, (SameAs, ReduceDims)) and tensor_decls:
                _resolve_input_ref(
                    rel.input_idx, tensor_decls, where=f"OutputDecl {decl.name!r}"
                )
            if (
                decl.dtype_from is not None
                and decl.dtype_from not in tensor_input_names
            ):
                raise ValueError(
                    f"OutputDecl {decl.name!r}: dtype_from={decl.dtype_from!r} is not a"
                    f" tensor input name. Known tensor inputs: {sorted(tensor_input_names)}"
                )

    # Extras (Numel / DimSize) only make sense against tensor inputs.
    for e in spec.extras:
        if e.input_name not in tensor_input_names:
            raise ValueError(
                f"ExtraArg {type(e).__name__}({e.input_name!r}) references unknown "
                f"tensor input. Known tensor inputs: {sorted(tensor_input_names)}"
            )

    geom = spec.geometry
    if geom is None:
        return

    if isinstance(geom, Elementwise):
        if not geom.block or any(b <= 0 for b in geom.block):
            raise ValueError(
                "Elementwise.block must be a non-empty tuple of positive ints, "
                f"got {geom.block!r}"
            )
        if geom.layout not in ("flat", "nd"):
            raise ValueError(
                f"Elementwise.layout must be 'flat' or 'nd', got {geom.layout!r}"
            )
        if geom.layout == "flat" and len(geom.block) != 1:
            raise ValueError(
                "Elementwise(layout='flat') requires block of length 1, got "
                f"block={geom.block!r}"
            )
        if len(geom.block) > 3:
            raise ValueError(
                f"Elementwise.block can have at most 3 dims, got {geom.block!r}"
            )
    elif isinstance(geom, Reduction):
        if geom.block_size <= 0:
            raise ValueError(f"Reduction.block_size must be > 0, got {geom.block_size}")
        if not geom.reduce_dims:
            raise ValueError("Reduction.reduce_dims must be non-empty")
    elif isinstance(geom, Custom):
        if not callable(geom.fn):
            raise ValueError("Custom.fn must be callable")
    else:
        raise TypeError(f"Unsupported Geometry: {geom!r}")
