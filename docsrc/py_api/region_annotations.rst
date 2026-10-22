.. _torch_tensorrt_region_annotations_py:

torch_tensorrt.region_annotations
================================

.. currentmodule:: torch_tensorrt.region_annotations

These experimental context managers choose how a selected group of model
operations executes. Use direct ``torch_tensorrt.compile(..., ir="dynamo",
strict=True, offload_module_to_cpu=False)`` capture with static shapes and an
eval-mode model. During ordinary eager execution, the annotated body runs
normally.

See :ref:`execute_in_torch` for the tutorial, supported settings, and save/load
limits. Runnable examples cover :ref:`eager and Inductor regions
<execute_in_torch_example>` and :ref:`automatic custom-kernel plugins
<fuse_in_plugin_example>`. The :ref:`agent-generated kernel example
<agentic_region_kernel_generation_example>` generates and evaluates a kernel
during compilation using a logged-in Codex or Claude CLI.

Region annotations
------------------

.. autofunction:: execute_in_torch

.. autofunction:: fuse_in_plugin

Kernel generation
-----------------

Pass ``generator=KernelGenerator(...)`` to ``fuse_in_plugin`` to generate a
kernel for the annotated PyTorch computation. ``provider`` chooses the author
(``claude`` or ``codex``); ``backend`` independently chooses the kernel language:
``triton``, NVIDIA ``cutile``, or ``cuda_cpp``. The latter generates CUDA C++ and
compiles it through CUDA Python's ``cuda.core`` and NVRTC; Numba is not a built-in
backend.

The compiler captures the region, loads the selected backend's packaged skill,
and asks the author for a structured proposal. Python workers compile candidates
into trial TensorRT engines, check their outputs and input immutability, and
measure their execution. The selected candidate must pass a final check using
fresh inputs before registration. Cached candidates receive fresh validation
before reuse. Inference executes the compiled plugin without invoking an agent.

Customize generation without editing the package::

    generator = KernelGenerator(
        provider="claude",
        backend="cutile",
        instructions="Prioritize coalesced loads for the broadcasted inputs.",
        instruction_files=("kernel-guidance.md",),
        objective="tensorrt",
        min_speedup=1.05,
    )

Instruction files supplement the shared skill and selected backend guide. Their
contents are snapshotted and included in cache identity, along with the backend
and schema versions, target, and evaluation policy. The author can request
bounded pages of named references and prior feedback through a structured
``read`` response. ``max_context_requests`` limits reads per proposal;
``author_timeout_s`` includes those reads. The controller retains responsibility
for compilation, checks, timing, and the overall ``timeout_s`` budget.

Choose the acceptance objective:

* ``candidate`` (default): select among correct generated candidates, without
  requiring a baseline speedup.
* ``eager``: require the specified improvement over the captured PyTorch region.
* ``tensorrt``: require the specified improvement over a fully compiled TensorRT
  version of that region.

For baseline objectives, ``min_speedup`` specifies the required baseline/candidate
latency ratio. Acceptance conservatively subtracts the baseline's median absolute
deviation (MAD) from its median, adds the candidate's MAD to its median, and
requires that adjusted ratio to meet the target. This is a noise allowance, not
a statistical confidence guarantee. Measurements use interleaved CUDA graph
samples controlled by ``benchmark_repeats``. Differences within measured noise
retain the incumbent. These are isolated-region measurements; measure the
complete model separately. Failure to satisfy correctness or the selected
objective raises ``RegionError``.

Generation currently requires a POSIX host and static, contiguous, nonempty
CUDA float32, float16, or bfloat16 tensor boundaries. Each region becomes one
kernel. Model requests include the captured graph and tensor metadata and use
the selected provider's account allowance. See the example for prerequisites and
the complete compilation workflow.

Skills and extension adapters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The author skill and backend references ship with the Python package. The
repository's ``.claude/skills/torch-tensorrt-kernel/SKILL.md`` is an interactive
entry point to the same instructions and controller. Inspect them without a
model call::

    python -m torch_tensorrt.dynamo.region_compilation._generation instructions --backend cuda_cpp
    python -m torch_tensorrt.dynamo.region_compilation._generation inspect PATH

The same module provides ``generate`` and ``evaluate`` commands for saved runs;
use ``--help`` for their arguments.

Advanced integrations can set either ``backend`` or ``provider`` to an explicit
``module:attribute`` import path. The module must be installed and importable in
worker subprocesses. A backend implements the internal
``_generation._backends.GenerationBackend`` protocol: ``name``, ``version``,
``proposal_schema``, ``instruction_resources``, ``target_metadata()``,
``validate_candidate()``, and ``make_registration()``. Its instructions may use
``package:relative/path.md`` resource names, and its registration must return the
existing ``KernelSpec`` or registration callback form.

A provider adapter is a callable or an object with ``propose()`` accepting the
keyword arguments ``model``, ``prompt``, ``workdir``, ``timeout_s``, ``schema``,
and ``evidence`` (or ``**kwargs``). It returns a JSON object under the supplied
response schema and runs in a bounded subprocess. These explicit adapters are
trusted Python code. Backend support requires a real compiler and registration
adapter; changing natural-language instructions alone does not add a language.

.. autoclass:: KernelGenerator

Errors
------

.. autoexception:: RegionError

.. autoexception:: RegionCaptureError
