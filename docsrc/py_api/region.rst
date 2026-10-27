.. _torch_tensorrt_region_py:

torch_tensorrt.region
=====================

.. currentmodule:: torch_tensorrt.region

These experimental context managers choose how a selected group of model
operations executes. Use direct ``torch_tensorrt.compile(..., ir="dynamo",
strict=True, offload_module_to_cpu=False)`` capture with static shapes and an
eval-mode model. During ordinary eager execution, the annotated body runs
normally.

See :ref:`execute_in_torch` for the tutorial, supported settings, and save/load
limits. Runnable examples cover :ref:`eager and Inductor regions
<execute_in_torch_example>`, :ref:`PyTorch custom operators <fused_op_example>`,
and :ref:`custom-kernel plugins <fused_kernel_example>`.
The :ref:`agent-generated kernel example <agentic_region_kernel_generation_example>`
generates and evaluates a kernel during compilation using a logged-in agent CLI.

Region annotations
------------------

.. autofunction:: execute_in_torch

.. autofunction:: fused_op

.. autofunction:: fused_kernel

Kernel generation
-----------------

Pass ``generator=KernelGenerator(...)`` to ``fused_kernel`` to generate a
Triton or cuTile kernel for the annotated PyTorch computation. Configure the
provider, backend, search budget, and cache through ``KernelGenerator``. The
compiler checks candidates against the captured computation and retains the
fastest passing candidate. Cached kernels are checked again before reuse.
Inference executes the compiled plugin without invoking an agent.

Generation currently requires a POSIX host and static, contiguous, nonempty
CUDA float32, float16, or bfloat16 tensor boundaries. Each region becomes one
kernel. Model requests include the captured graph and tensor metadata and use
the selected CLI's account allowance. See the example for prerequisites and
the complete compilation workflow.

.. autoclass:: KernelGenerator

Errors
------

.. autoexception:: RegionError

.. autoexception:: RegionCaptureError
