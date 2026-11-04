.. _torch_tensorrt_region_py:

torch_tensorrt.region
=====================

.. automodule:: torch_tensorrt.region

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

Pass ``generator=KernelGenerator(...)`` to ``fused_kernel`` to generate a kernel
for the annotated computation during compilation. ``provider`` chooses the
author (``claude`` or ``codex`` through their logged-in CLIs); ``backend``
independently chooses the kernel language: ``triton``, NVIDIA ``cutile``, or
``cuda_cpp`` (CUDA C++ compiled through CUDA Python's ``cuda.core`` and NVRTC).
Construct the generator before compilation, for example in ``__init__``; it
cannot be created inside the traced ``forward``.

The author receives the captured graph, its tensor contract, the packaged
backend guide, any custom instructions, and measured feedback. Torch-TensorRT
compiles each proposal into a trial TensorRT engine, checks its outputs and
input immutability, rejects writes outside its tensors, and measures it. The
selected kernel passes a final check on fresh inputs before it is registered,
and inference runs it inside the compiled engine without an agent::

    generator = KernelGenerator(
        provider="claude",
        backend="cutile",
        instructions="Prioritize coalesced loads for the broadcasted inputs.",
        instruction_files=("kernel-guidance.md",),
        objective="tensorrt",
        min_speedup=1.05,
    )

``SearchPolicy`` controls exploration independently of the provider::

    generator = KernelGenerator(
        max_attempts=4,
        search=SearchPolicy(
            batch_size=2,
            max_no_improvement=2,
            finalists=2,
            resume=True,
            tuning={"num_warps": (4, 8)},
        ),
        diagnostics=True,
    )

Each author round proposes up to ``batch_size`` candidates, and the controller
expands each into tuning variants, starting from the authored launch settings
and moving outward through the declared choices. ``max_candidates`` bounds all
search evaluations; final validation and cache checks are outside it and have
time reserved within ``timeout_s``. Unfinished searches are checkpointed, and
``resume=True`` remeasures their retained candidates. The class reference below
describes every option.

Acceptance
~~~~~~~~~~

* ``objective="candidate"`` (default) selects the fastest correct candidate.
  ``"eager"`` and ``"tensorrt"`` also require ``min_speedup`` over the captured
  PyTorch region or a TensorRT engine built from it with your engine build
  settings, after allowing for measured timing noise.
* Without explicit ``rtol``/``atol``, outputs are compared with a float64
  evaluation of the reference, allowing a small multiple of eager PyTorch's own
  rounding error. Reductions and matrix products therefore need not reproduce
  PyTorch's summation order, while elementwise results stay tight.
* Timings are interleaved CUDA graph samples of the isolated region. They do
  not establish a whole-model speedup, and passing checks do not prove
  correctness for every input.

The persistent cache, under ``$XDG_CACHE_HOME/torch_tensorrt/region_kernels`` by
default, keys accepted kernels by the captured computation, tensor contract,
backend toolchain, instruction contents, compilation settings, and evaluation
policy. A cached kernel is revalidated before reuse, and identical regions in
one compilation share one kernel. Dry runs (``dryrun=True``) do not invoke the
agent; they show the region executing its PyTorch reference.

Generation requires a POSIX host and static, contiguous, nonempty float16,
bfloat16, or float32 CUDA tensor boundaries; each region becomes one kernel.
Generated code and extension adapters run locally. Worker subprocesses contain
timeouts and crashes; they are not a sandbox for hostile code.

Inspecting and replaying runs (experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each search saves its task, exact instruction snapshot, proposals, and
feedback. The following commands use private module paths that may change::

    python -m torch_tensorrt.dynamo.region_compilation._generation instructions --backend cuda_cpp
    python -m torch_tensorrt.dynamo.region_compilation._generation inspect PATH
    python -m torch_tensorrt.dynamo.region_compilation._generation evaluate PATH --candidate CANDIDATE.json
    python -m torch_tensorrt.dynamo.region_compilation._generation generate PATH --resume --batch-size 2

``instructions`` prints the packaged guidance and the backend's proposal schema
without a model call. ``inspect`` accepts a run, an attempt, a cache entry, or
the artifact path printed in errors. ``generate`` replays a saved task with its
recorded search policy, budgets, and exact instruction snapshot unless options
override them, and uses the run's own cache root unless ``--cache-dir`` names
another, so an accepted kernel is revalidated rather than searched for again;
see ``--help``. The repository's
``.claude/skills/torch-tensorrt-kernel/SKILL.md`` drives the same commands from
Claude Code.

Extension adapters (experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``provider`` and ``backend`` also accept an explicit ``module:attribute`` path to
trusted Python code that is importable in worker subprocesses. A provider adapter
is a callable, an object with ``propose()``, or a class constructed without
arguments. It receives the keyword arguments ``model``, ``prompt``, ``workdir``,
``timeout_s``, ``schema``, and ``evidence`` and returns a JSON object with
``action`` (``propose``, ``read``, ``diagnose``, or ``stop``), ``candidates``,
``resource``, and ``offset``. Raising ``TimeoutError`` costs only the current
author turn; any other exception ends the search.

A backend implements the internal ``_generation._backends.GenerationBackend``
protocol: ``name``, ``version``, ``proposal_schema``, ``instruction_resources``,
``target_metadata()``, ``validate_candidate()``, and ``make_registration()``,
which returns a ``KernelSpec`` or a ``functools.partial`` of
``torch_tensorrt.kernels.triton_op``, ``cutile_op``, or ``ptx_op``. An optional
``launch()`` method enables the out-of-bounds write check. Instruction resources
may use ``package:relative/path.md`` names. Natural-language instructions alone
cannot add a kernel language.

.. autoclass:: KernelGenerator

.. autoclass:: SearchPolicy

Errors
------

.. autoexception:: RegionError

.. autoexception:: RegionCaptureError
