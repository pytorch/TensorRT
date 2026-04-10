"""AOT kernel backend implementations for the TTA custom plugin system.

Each sub-module implements the ``aot_impl_<backend>`` function that drives a
backend-specific compile pipeline and returns the QDP AOT 4-tuple::

    (kernel_name, code_bytes, KernelLaunchParams, SymIntExprs)

Backends
--------
_triton   — Triton JIT kernels compiled to PTX via triton.compile()
_cutile   — cuTILE programs compiled to CUBIN via cuda.tile compile_tile()
_cutedsl  — CuTe DSL @cute.jit kernels compiled via cutlass.cute.compile()

All three backends share ``_qdp_utils`` helpers for sandboxing, launch-arg
analysis, artifact dumping, and the unified ``AOTMetadata`` result type.
"""
