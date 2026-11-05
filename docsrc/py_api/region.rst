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
Construct the generator outside the traced ``forward``, for example once at
module level as the example does, so the model's constructor need not change.
Eager and training runs execute the annotated block as written; only
``torch_tensorrt.compile`` generates and installs a kernel.

The author receives the captured graph, its tensor contract, the packaged
backend guide, any custom instructions, and measured feedback. Torch-TensorRT
checks each proposal's outputs and input immutability, rejects writes outside
its tensors, and measures it, by default launching the kernel directly. The
selected kernel passes a final check on fresh inputs inside a trial TensorRT
engine before it is registered, and inference runs it inside the compiled
engine without an agent::

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

Validating on real inputs
~~~~~~~~~~~~~~~~~~~~~~~~~

Synthetic inputs cannot cover every value a model produces, such as the
``-inf`` entries of masked attention scores. ``validation_inputs`` adds real
data: a reusable iterable of model input batches. Each batch is a tensor or a
tuple of tensors for the positional inputs given to ``torch_tensorrt.compile``,
followed by a dict of the keyword inputs if it was given ``kwarg_inputs``; a
dict alone holds only keyword inputs. A ``DataLoader`` works::

    loader = torch.utils.data.DataLoader(dataset, batch_size=8, drop_last=True)
    generator = KernelGenerator(validation_inputs=loader)

Compilation runs the model, with its own weights, on the first 8 batches, or
fewer once the tensors kept from them reach 1 GiB, and keeps the tensors that
reach each generated region. Every candidate, final validation, and cache
check is then also checked on them, as cases named ``sample 1``, ``sample 2``,
and so on. Regions are compiled for static shapes, so a batch must match the
compiled input shapes and dtypes (a ``DataLoader`` may need
``drop_last=True``); otherwise compilation fails before any agent call.
Batches are moved to the compiled inputs' device.

Each run saves the kept tensors as ``samples.pt``, a list with one list of
region input tensors per batch, which loads with
``torch.load(weights_only=True)``; a run that reuses a cached kernel deletes
it after the check. The tensors are not part of the cache key: a cached
kernel, or the kernel of an identical region in the same compilation, is
reused once it passes a check on the current tensors, so new weights or other
batches do not by themselves start a search. For an identical region that
check covers correctness only, since the shared kernel already met the
objective in this compilation. A region whose tensors fail it searches again,
on its own tensors and those of the identical regions before it, and every
identical region then uses the new kernel. Failure feedback sent to the author
can include values from these tensors.

Runs that search, and refused cache-only runs, keep their ``samples.pt``, and
runs are never pruned. To reclaim space, delete a task directory's ``run-*``
directories other than the one its ``accepted.json`` names (all of them for a
task compiled with ``use_cache=False``), while no compilation from that cache is
running.

Acceptance
~~~~~~~~~~

* ``objective="candidate"`` (default) selects the fastest correct candidate.
  ``"eager"`` and ``"tensorrt"`` also require ``min_speedup`` over the captured
  PyTorch region or a TensorRT engine built from it with your engine build
  settings, after allowing for measured timing noise.
* Search evaluations screen candidates and tuning variants
  (``SearchPolicy(screening="launch")``, the default): each candidate is
  compiled and registered as the plugin an engine would embed, which applies
  the plugin's own checks, such as the shared memory a TensorRT plugin launch
  allows a Triton kernel, and is then launched directly, which saves a
  TensorRT engine build per evaluation. Finalists and cached kernels are
  always checked again as plugins inside a trial engine, and only those engine
  measurements decide ``"eager"`` and ``"tensorrt"``; screened timings only
  rank candidates. A problem that only an engine build reveals therefore
  surfaces in final validation, which tries up to ``finalists`` candidates.
  Under ``"tensorrt"``, screens still build the TensorRT baseline until one
  search evaluation has built it, so a region that TensorRT cannot build stops
  the search after its first evaluation. ``screening="engine"`` builds an
  engine, and for ``"tensorrt"`` a baseline engine, for every evaluation.
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
policy. A cached kernel is revalidated before reuse,
and identical regions in one compilation all use one kernel. Dry runs
(``dryrun=True``) do not invoke the agent; they show the region executing its
PyTorch reference.

Generation requires a POSIX host and static, contiguous, nonempty float16,
bfloat16, or float32 CUDA tensor boundaries; each region becomes one kernel.
Generated code and extension adapters run locally. Worker subprocesses contain
timeouts and crashes; they are not a sandbox for hostile code.

Cache-only compilation
~~~~~~~~~~~~~~~~~~~~~~

CI and production builds can forbid generation, so that compilation never
calls an agent: pass ``allow_generation=False``, or set the environment
variable ``TORCH_TENSORRT_REGION_GENERATION=0`` to apply it to every generator
without code changes (any value other than ``0`` or ``1`` is an error).
Compilation then reuses a cached kernel after revalidating it, and a cache
miss or a failed revalidation raises ``RegionError`` without calling the
provider. Generate the kernel first by compiling once with generation allowed,
on the same GPU type with the same toolchain and settings,
and give the cache-only build that ``cache_dir``.

The ``cache_dir`` must exist, and it must be owned by and writable by the user
that compiles: every compilation records a run there, even one that reuses a
cached kernel, and a directory owned by another user is refused. When you copy
a cache into an image, ``chown`` it to the user that runs the build. A
cache-only build never creates a missing ``cache_dir``.

Revalidation repeats the final check, so under ``objective="eager"`` or
``"tensorrt"`` it also remeasures the speedup, which a busy GPU can push below
``min_speedup``. The error explains a refusal: after a cache miss it names the
``task.json`` fields in which a kernel cached for the same captured computation
differs, such as ``target`` after a GPU or toolchain change; after a failed
revalidation it gives the stage and error, or the remeasured speedup, and the
run keeps the full report in ``cache_check.json``. A revalidation that could
not complete, at stage ``budget``, ``setup``, ``timeout``, or ``baseline``,
does not show that the kernel is wrong: retry it, for example on a less busy
GPU or with larger ``timeout_s`` and ``candidate_timeout_s``, before generating
a new kernel. A kernel refused only for its remeasured speedup is correct, so
the error suggests retrying on a less busy GPU before searching for a faster
one.

Data and trust
~~~~~~~~~~~~~~

* The provider receives the region's captured graph code, its non-tensor
  constants, tensor shapes and dtypes, the instructions, and measured feedback,
  and runs under your CLI login and account. Tensor values are not sent, except
  that failure feedback can quote values from ``validation_inputs`` samples.
* Runs and the cache are created readable only by you. A cache directory owned
  by another user is refused, and one that others can write is reported,
  because a planted kernel would be revalidated and then loaded.
* Runs store the reference as JSON and validation samples as plain tensors
  that load with ``torch.load(weights_only=True)``, so loading a run executes
  no code from it. The reference can call only ATen and prims operators and
  Python's indexing and arithmetic operators (``getitem``, ``add``, ``sub``,
  ``mul``, ``truediv``, ``floordiv``, ``neg``). Loading refuses operators that
  PyTorch tags as random, Inductor's internal primitives, operators that access
  files, and operators that declare they mutate their arguments. Kernel source
  from a run is validated before it is loaded.
* These checks are not a sandbox: replaying or evaluating a run still executes
  its reference and kernel source, and PyTorch operators do not check every
  argument that a crafted reference can pass them. Replay a run only if you
  trust its source.

Inspecting and replaying runs (experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generation itself runs inside ``torch_tensorrt.compile``. These commands are a
debugging aid that inspects, re-evaluates, or replays saved runs. Each search
saves its task, exact instruction snapshot, proposals, and feedback. The
commands use private module paths that may change::

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
``launch()`` method enables the out-of-bounds write check and launch screening,
which checks and times candidates through it: it must run the kernel its
plugin runs, with the same launch settings, and be safe to capture in a CUDA
graph, and ``make_registration()`` should reject candidates that the launch
can run but the plugin cannot. Otherwise use ``SearchPolicy(screening="engine")``.
Instruction resources may use ``package:relative/path.md`` names.
Natural-language instructions alone cannot add a kernel language.

.. autoclass:: KernelGenerator

.. autoclass:: SearchPolicy

Errors
------

.. autoexception:: RegionError

.. autoexception:: RegionCaptureError
