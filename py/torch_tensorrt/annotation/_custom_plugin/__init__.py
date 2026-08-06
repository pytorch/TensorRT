"""Custom plugin sub-package: QDP-backed plugin descriptor and lowering.

This sub-package bridges user-supplied GPU kernels (Triton, cuTILE, CuTe DSL)
to TensorRT's Quickstart Dynamic Plugin (QDP) framework, enabling annotated
boundary ops to be lowered to first-class ``IPluginV3`` layers at TRT engine
compile time.

Role in the compilation pipeline
---------------------------------
1. **Annotation** — the user calls ``tta.custom_plugin(kernel, meta_impl=...)``
   (re-exported here as ``custom_plugin``), which returns a
   ``CustomPluginSpec`` that is stored in the boundary op's
   ``AnnotationMetadata``.
2. **Registration** — at compile time, no-weights plugins go through
   ``trt_plugins.custom_op(op_name, impl=spec)``: the desc + JIT impl are
   provided by upstream ``generate_plugin`` (schema-derived), and the
   ``_aot_register`` hook adds ``@trtp.autotune`` + ``@trtp.aot_impl`` via
   ``register_autotune_and_aot``. Weighted plugins continue to use
   ``register_custom_plugin`` (registers the desc/autotune/aot trio with
   weights counted toward ``num_inputs``).
3. **Lowering** — ``lower_custom_plugin_descriptor`` calls ``trtp.op.<ns>.<name>``
   to insert an ``IPluginV3`` layer into the ``INetworkDefinition``.

Public surface
--------------
``CustomPluginSpec``
    Dataclass returned by ``custom_plugin()``.  Carries the op name, kernel
    specs, meta-shape implementation, and optional tactic table.

``custom_plugin``
    Factory that builds a ``CustomPluginSpec`` from a kernel spec, auto-
    computing a deterministic QDP op name from the kernel fingerprint.

``lower_custom_plugin_descriptor``
    Converts a ``CustomPluginSpec`` into a TRT ``IPluginV3`` layer and
    returns the output ``trt.ITensor`` (or tuple thereof).

``register_custom_plugin``
    Registers the full ``@trtp.register`` / ``@trtp.autotune`` /
    ``@trtp.aot_impl`` trio for a descriptor's op name. Used by the
    weighted-plugin path (where the desc must include weight inputs).
    Idempotent at the process level.

``register_autotune_and_aot``
    Layers ``@trtp.autotune`` + ``@trtp.aot_impl`` on top of a desc already
    registered by upstream ``generate_plugin``. Used by the no-weights path
    via ``custom_op(..., _aot_register=...)``. Idempotent at the process
    level.

``QDPRuntimeError``
    Raised when TRT's QDP framework encounters a runtime error (e.g. shape
    mismatch, unsupported dtype) during plugin execution.

``TTAPluginError``
    Raised for TTA-level plugin configuration errors (e.g. missing meta_impl,
    invalid kernel spec).

``SymbolicTensor`` / ``TensorRole``
    Proxy used during AOT kernel compilation to carry symbolic shape
    expressions.  ``TensorRole`` distinguishes input from output tensors so
    that ``analyze_launch_args`` can reconstruct the correct QDP binding
    indices.
"""

# ---------------------------------------------------------------------------
# Re-exports from sub-modules
# ---------------------------------------------------------------------------

from ._descriptor import (
    CustomPluginSpec,
    custom_plugin,
    lower_custom_plugin_descriptor,
    register_autotune_and_aot,
    register_custom_plugin,
)
from ._qdp_utils import QDPRuntimeError, TTAPluginError
from ._symbolic import SymbolicTensor, TensorRole

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "CustomPluginSpec",
    "custom_plugin",
    "lower_custom_plugin_descriptor",
    "register_autotune_and_aot",
    "register_custom_plugin",
    "QDPRuntimeError",
    "TTAPluginError",
    "SymbolicTensor",
    "TensorRole",
]
