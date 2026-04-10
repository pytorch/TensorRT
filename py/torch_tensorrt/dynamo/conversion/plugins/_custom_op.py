from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
from torch.fx.node import Node

from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._ConverterRegistry import ConverterPriority
from torch_tensorrt.dynamo.conversion.plugins._generate_plugin import (
    _probe_num_outputs,
    generate_plugin,
)
from torch_tensorrt.dynamo.conversion.plugins._generate_plugin_converter import (
    generate_plugin_converter,
)

if TYPE_CHECKING:
    from torch_tensorrt.annotation._custom_plugin._descriptor import CustomPluginSpec


def custom_op(
    op_name: str,
    impl: Optional["CustomPluginSpec"] = None,
    capability_validator: Optional[Callable[[Node, CompilationSettings], bool]] = None,
    priority: ConverterPriority = ConverterPriority.STANDARD,
    supports_dynamic_shapes: bool = False,
    requires_output_allocator: bool = False,
    *,
    use_aot_if_available: bool = True,
    _aot_register: Optional[Callable[[], None]] = None,
) -> None:
    """
    Generate the Plugin and corresponding Plugin Converter using external kernels
    and TensorRT Quick Deployable Plugin APIs.

    Args:
        op_name: the plugin name in ``"namespace::name"`` form.  A matching
            ``torch.library.custom_op`` must already exist (or be auto-created
            via ``impl``).
        impl: optional ``tta.CustomPluginSpec`` from ``tta.custom_plugin(...)``.
            When provided:

            1. The torch op is auto-registered (``impl.auto_register_torch_op``).
            2. TRT QDP descriptors are registered via ``register_custom_plugin``
               which handles both activation and weight tensor inputs, then adds
               ``@trtp.autotune`` and ``@trtp.aot_impl`` for AOT kernel dispatch.
            3. A Dynamo converter is registered via ``generate_plugin_converter``
               (the same path as the JIT plugin) when there are no weights.
               For weighted plugins, a minimal custom converter is registered
               that injects weight tensors as ``trt.add_constant`` layers before
               delegating to ``impl.lower_to_trt``.

            When ``None`` (default) the existing JIT path (``generate_plugin`` +
            ``generate_plugin_converter``) is used unchanged.
        capability_validator: optional node capability predicate. A lambda that
            takes a ``torch.fx.Node`` and determines if the converter can handle
            it; nodes that fail this predicate run in PyTorch.
        priority: converter registry priority.
        supports_dynamic_shapes: whether the converter supports dynamic shapes.
        requires_output_allocator: whether the converter requires an output
            allocator (e.g. data-dependent operators).
        use_aot_if_available: forwarded to ``generate_plugin_converter``; when
            ``True`` (the default), the converter prefers the AOT impl if the op
            has one registered.
        _aot_register: internal hook used by ``torch_tensorrt.kernels`` to
            register a cuda-python AOT impl between the plugin descriptor and
            the converter. Not part of the public API; pass ``None`` (the
            default) for ordinary use.
    """
    if impl is not None:
        from torch_tensorrt.annotation._custom_plugin._descriptor import (
            register_custom_plugin,
        )

        impl.auto_register_torch_op(op_name)

        namespace, op_local_name = op_name.split("::")
        torch_op = getattr(getattr(torch.ops, namespace), op_local_name)
        schema = torch_op._schemas[""]
        n_tensor_inputs = sum(
            1 for a in schema.arguments
            if a.type.isSubtypeOf(torch._C.TensorType.get())
        )

        # QDP registration: includes weight tensors in the input count so TRT
        # can receive them as trt.add_constant outputs.
        # Probe torch_op (already registered by auto_register_torch_op above)
        # to get the correct output count — same mechanism as generate_plugin.
        num_outputs = _probe_num_outputs(torch_op, schema)
        register_custom_plugin(
            impl,
            num_inputs=n_tensor_inputs + len(impl.weights),
            num_outputs=num_outputs,
            qdp_name=op_name,
        )

        if not impl.weights:
            # No weights: reuse generate_plugin_converter, the same path as the
            # JIT plugin.  After register_custom_plugin, the op is in QDP_REGISTRY
            # with aot_impl registered, so generate_plugin_converter detects that
            # and sets aot=True automatically — no separate converter code needed.
            generate_plugin_converter(
                op_name,
                capability_validator,
                priority,
                supports_dynamic_shapes,
                requires_output_allocator,
                use_aot_if_available=use_aot_if_available,
            )
        else:
            # Weighted plugins need a custom converter: generate_plugin_converter
            # cannot inject weight tensors as trt.add_constant before the plugin
            # call because it reads input count from plugin.input_tensor_names,
            # which includes weights, but the FX node only carries the activation
            # args.  lower_to_trt handles the weight injection via
            # lower_custom_plugin_descriptor.
            from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
                dynamo_tensorrt_converter,
            )
            from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor

            torch_overload = getattr(torch_op, "default")
            _impl = impl
            _n_act = n_tensor_inputs
            _qdp_name = op_name

            def _impl_converter(
                ctx: Any,
                target: Any,
                args: Any,
                kwargs: Any,
                name: str,
            ) -> Any:
                unique_id = uuid.uuid4()
                itensor_args = [
                    get_trt_tensor(ctx, t, f"inp{i}_{unique_id}")
                    for i, t in enumerate(args[:_n_act])
                ]
                return _impl.lower_to_trt(ctx, itensor_args, name, qdp_name=_qdp_name)

            dynamo_tensorrt_converter(
                torch_overload,
                capability_validator=capability_validator,
                priority=priority,
                supports_dynamic_shapes=supports_dynamic_shapes,
                requires_output_allocator=requires_output_allocator,
            )(_impl_converter)
    else:
        generate_plugin(op_name)
        if _aot_register is not None:
            _aot_register()
        generate_plugin_converter(
            op_name,
            capability_validator,
            priority,
            supports_dynamic_shapes,
            requires_output_allocator,
            use_aot_if_available=use_aot_if_available,
        )
