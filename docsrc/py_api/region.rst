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
kernel for the annotated PyTorch computation. ``provider`` chooses the author
(``claude`` or ``codex``); ``backend`` independently chooses the kernel language:
``triton``, NVIDIA ``cutile``, or ``cuda_cpp``. The latter generates CUDA C++ and
compiles it through CUDA Python's ``cuda.core`` and NVRTC; Numba is not a built-in
backend.

The compiler captures the region, loads the selected backend's packaged skill,
and asks the author for structured proposals. Python workers compile candidates
into trial TensorRT engines, check their outputs and input immutability, and
measure their execution. The selected candidate must pass a final check using
fresh inputs before registration; an optional shortlist supplies alternatives if
the first candidate fails. Cached candidates receive fresh validation
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

Use ``SearchPolicy`` to control exploration independently of the provider::

    generator = KernelGenerator(
        provider="claude",
        backend="triton",
        max_attempts=4,
        search=SearchPolicy(
            batch_size=2,
            max_candidates=16,
            max_no_improvement=2,
            max_variants=4,
            finalists=2,
            resume=True,
            tuning={"num_warps": (4, 8)},
        ),
        diagnostics=True,
    )

``max_attempts`` bounds author rounds; ``batch_size`` bounds proposals per round.
``max_candidates`` limits actual search evaluations, including generated tuning
variants, resumed candidates, and diagnostic evaluations. Its default is
``max_attempts * batch_size``. ``max_no_improvement`` stops after that many
complete author rounds without a better correct candidate, including progress
toward a speedup target not yet reached. Final validation and cache rechecks
are outside this candidate cap, with evaluation time reserved inside the
search timeout. Final registration then follows the normal compilation path.

``tuning`` declares allowed values for backend launch fields, or named constants
such as ``constants.BLOCK`` when the backend supports them. The controller
generates at most ``max_variants`` variants per authored candidate, and each
variant receives the same source and correctness checks. Source and kernel
names are not tuning knobs. ``finalists`` bounds the number of shortlisted
candidates eligible for fresh final validation; if one fails, the controller
can try the next.

Tuning forms a bounded Cartesian product of the supplied choices. Changing a
constant does not automatically recompute the launch grid; supply compatible
settings. Every configured knob is replaced with an allowed value before the
backend validates and builds the candidate.

Unfinished searches are checkpointed automatically. ``resume=True`` restores
compatible prior search evidence and remeasures retained candidates, even if
the original run did not enable resume. Each invocation grants fresh time,
round, and evaluation budgets. Final validation reserves time for the configured alternatives and
checks them sequentially. ``diagnostics=True`` enables actual compiler and
execution facts in worker feedback; unavailable diagnostics remain explicitly
unavailable.

Instruction files supplement the shared skill and selected backend guide. Their
contents are snapshotted and included in cache identity, along with the backend
and schema versions, target, and evaluation policy. The author can request
bounded pages of named references and prior feedback through a structured
``read`` response. With ``diagnostics=True``, the author may also request
``diagnose`` on the current best candidate. It can return ``stop`` when further
proposals are not useful.
``max_context_requests`` bounds read and diagnostic requests per author round;
``author_timeout_s`` includes those requests. The controller retains responsibility
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
``generate`` preserves the saved search policy and diagnostic setting unless
explicit flags override them. For example::

    python -m torch_tensorrt.dynamo.region_compilation._generation generate PATH --resume --batch-size 2 --max-candidates 16 --tuning '{"num_warps":[4,8]}' --finalists 2

Use ``--no-resume`` or ``--no-diagnostics`` to turn off a saved setting, and
``--tuning '{}'`` to clear saved tuning domains. Resuming uses the saved run's
cache root unless ``--cache-dir`` overrides it. ``inspect`` also displays the
saved checkpoint and current run's search summary when present. With the same
backend and no explicit guidance override, replay uses the exact saved
instruction bundle, including its captured built-in guides. This preserves
instruction identity even if an original guidance file moved or changed. The
bundle's contents and hashes must match the saved task. See
``--help`` for all options.

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
response schema and runs in a bounded subprocess. Author responses support a
single candidate, a candidate batch, evidence reads, an incumbent diagnostic
request, or stopping exploration. Existing custom providers returning a direct
single-candidate object remain supported. These explicit adapters are
trusted Python code. Backend support requires a real compiler and registration
adapter; changing natural-language instructions alone does not add a language.

.. autoclass:: KernelGenerator

.. autoclass:: SearchPolicy

Errors
------

.. autoexception:: RegionError

.. autoexception:: RegionCaptureError
