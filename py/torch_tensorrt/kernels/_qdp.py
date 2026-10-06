# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility with TensorRT QDP's private registries and shape expressions.

Keep version-dependent access here. Registration's lock and Torch lifetime
management belong to the caller; these helpers never compile kernels.
"""

import logging
from typing import Any, NamedTuple

import torch

_LOGGER = logging.getLogger(__name__)
_MISSING_REGISTRATION = object()


def _patch_trt_shape_expr_reflected_ops() -> None:
    # TODO(upstream-trt): trtp.ShapeExpr defines forward __mul__ / __add__
    # but not the reflected __rmul__ / __radd__. torch_tensorrt lowers meta-fn
    # shape expressions via sympy.lambdify(..., "math"), which emits canonical
    # forms like ``lambda N: 2*N`` — at runtime Python does ``int * ShapeExpr``,
    # falls back to ``ShapeExpr.__rmul__``, and crashes with TypeError.
    # Reflected forms are commutative so aliasing fwd -> rev is safe.
    #
    # File against NVIDIA/TensorRT (the trtp Python plugin module). Once
    # trtp.ShapeExpr ships __rmul__ / __radd__ natively, this whole function
    # becomes a no-op (the ``not hasattr(cls, rev)`` guard self-disables) and
    # can be deleted. It runs only while registering a plugin.
    try:
        import tensorrt.plugin as trtp
    except ImportError:
        return
    cls = getattr(trtp, "ShapeExpr", None)
    if cls is None:
        return
    for fwd, rev in (("__mul__", "__rmul__"), ("__add__", "__radd__")):
        if hasattr(cls, fwd) and not hasattr(cls, rev):
            try:
                setattr(cls, rev, getattr(cls, fwd))
            except (AttributeError, TypeError):
                pass


class _QDPRegistrationState(NamedTuple):
    """Process-global state that must survive a failed registration unchanged."""

    qdp_definition: Any
    qdp_creator: Any
    native_creator: Any
    op_namespace: Any
    op_definition: Any
    converter_target: Any
    converter_entry: Any


def _snapshot_qdp_registration(op_name: str) -> _QDPRegistrationState:
    """Capture the exact slots ``custom_op`` may mutate for ``op_name``."""
    import tensorrt as trt
    import tensorrt.plugin as trtp
    from tensorrt.plugin._lib import QDP_CREATORS, QDP_REGISTRY

    from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
        DYNAMO_ATEN_CONVERTERS,
    )

    namespace, name = op_name.split("::")
    op_namespace = getattr(trtp.op, namespace, _MISSING_REGISTRATION)
    op_definition = (
        getattr(op_namespace, name, _MISSING_REGISTRATION)
        if op_namespace is not _MISSING_REGISTRATION
        else _MISSING_REGISTRATION
    )
    try:
        converter_target = getattr(getattr(torch.ops, namespace), name).default
    except AttributeError:
        converter_target = _MISSING_REGISTRATION
    converter_entry = (
        DYNAMO_ATEN_CONVERTERS.get(converter_target, _MISSING_REGISTRATION)
        if converter_target is not _MISSING_REGISTRATION
        else _MISSING_REGISTRATION
    )
    # Converter registration appends to an existing list in place. Preserve a
    # shallow copy rather than the mutable list itself so rollback can restore
    # the pre-call contents.
    if isinstance(converter_entry, list):
        converter_entry = list(converter_entry)

    return _QDPRegistrationState(
        QDP_REGISTRY.get(op_name, _MISSING_REGISTRATION),
        QDP_CREATORS.get(op_name, _MISSING_REGISTRATION),
        trt.get_plugin_registry().get_creator(name, "1", namespace),
        op_namespace,
        op_definition,
        converter_target,
        converter_entry,
    )


def _rollback_qdp_registration(op_name: str, state: _QDPRegistrationState) -> None:
    """Best-effort rollback of QDP and converter state created after ``state``.

    TensorRT exposes no transaction around ``register`` / ``impl`` /
    ``aot_impl``. Restore each Python registry slot and use its public native
    creator deregistration API. Every removal is conditioned on the slot being
    absent in the snapshot, so a failed attempt never deletes pre-existing
    state.
    """
    import tensorrt as trt
    import tensorrt.plugin as trtp
    from tensorrt.plugin._lib import QDP_CREATORS, QDP_REGISTRY

    from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
        DYNAMO_ATEN_CONVERTERS,
    )

    namespace, name = op_name.split("::")

    try:
        if state.converter_target is not _MISSING_REGISTRATION:
            if state.converter_entry is _MISSING_REGISTRATION:
                DYNAMO_ATEN_CONVERTERS.pop(state.converter_target, None)
            else:
                DYNAMO_ATEN_CONVERTERS[state.converter_target] = state.converter_entry
    except Exception:
        _LOGGER.warning(
            "Could not roll back the converter for %s", op_name, exc_info=True
        )

    current_definition = QDP_REGISTRY.get(op_name, _MISSING_REGISTRATION)
    try:
        current_namespace = getattr(trtp.op, namespace, _MISSING_REGISTRATION)
        if state.op_definition is _MISSING_REGISTRATION:
            if current_namespace is not _MISSING_REGISTRATION and hasattr(
                current_namespace, name
            ):
                current_op_definition = getattr(current_namespace, name)
                # A namespace can be shared by many plugins. Only remove the
                # attribute installed by this attempt, identified by the same
                # PluginDef object stored in QDP_REGISTRY.
                if (
                    current_definition is _MISSING_REGISTRATION
                    or current_op_definition is current_definition
                ):
                    delattr(current_namespace, name)
        elif current_namespace is _MISSING_REGISTRATION:
            setattr(trtp.op, namespace, state.op_namespace)
            setattr(state.op_namespace, name, state.op_definition)
            current_namespace = state.op_namespace
        else:
            setattr(current_namespace, name, state.op_definition)

        if (
            state.op_namespace is _MISSING_REGISTRATION
            and current_namespace is not _MISSING_REGISTRATION
            and not any(
                key != "_namespace" and not key.startswith("__")
                for key in vars(current_namespace)
            )
        ):
            delattr(trtp.op, namespace)
    except Exception:
        _LOGGER.warning("Could not roll back trtp.op for %s", op_name, exc_info=True)

    try:
        if state.qdp_definition is _MISSING_REGISTRATION:
            QDP_REGISTRY.pop(op_name, None)
        else:
            QDP_REGISTRY[op_name] = state.qdp_definition
    except Exception:
        _LOGGER.warning(
            "Could not roll back the QDP descriptor for %s", op_name, exc_info=True
        )

    try:
        plugin_registry = trt.get_plugin_registry()
        current_native_creator = plugin_registry.get_creator(name, "1", namespace)
        # An exact-name native creator cannot coexist with another creator. If
        # there was one before the attempt, it is necessarily pre-existing and
        # must be retained even if QDP_CREATORS temporarily pointed at it.
        if state.native_creator is None and current_native_creator is not None:
            if not plugin_registry.deregister_creator(current_native_creator):
                _LOGGER.warning(
                    "TensorRT refused to deregister the native QDP creator for %s; "
                    "future registration attempts will remain fail-closed",
                    op_name,
                )
    except Exception:
        _LOGGER.warning(
            "Could not deregister the native QDP creator for %s", op_name, exc_info=True
        )

    try:
        if state.qdp_creator is _MISSING_REGISTRATION:
            QDP_CREATORS.pop(op_name, None)
        else:
            QDP_CREATORS[op_name] = state.qdp_creator
    except Exception:
        _LOGGER.warning(
            "Could not restore the QDP creator map for %s", op_name, exc_info=True
        )


def assert_qdp_name_available(op_name: str) -> None:
    import tensorrt as trt
    import tensorrt.plugin as trtp
    from tensorrt.plugin._lib import QDP_CREATORS, QDP_REGISTRY

    namespace, name = op_name.split("::")
    op_namespace = getattr(trtp.op, namespace, None)
    native_creator = trt.get_plugin_registry().get_creator(name, "1", namespace)
    if (
        op_name in QDP_REGISTRY
        or op_name in QDP_CREATORS
        or (op_namespace is not None and hasattr(op_namespace, name))
        or native_creator is not None
    ):
        raise ValueError(
            f"plugin '{op_name}' is already registered with TensorRT QDP; choose "
            "a unique qualified name."
        )
