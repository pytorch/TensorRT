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
independently chooses the kernel language: ``triton``, NVIDIA ``cutile``,
``cuda_cpp`` (CUDA C++ compiled through CUDA Python's ``cuda.core`` and NVRTC),
or NVIDIA ``cutedsl`` (CuTe DSL).
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

.. autoclass:: KernelGenerator

.. autoclass:: SearchPolicy

What a compilation generated or loaded
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every kernel the generator registers in a process, newly generated, loaded
from the cache or shared by identical regions, is recorded with its region,
task and kernel digests, provider, generation time and the region-level
measurement from its ``accepted.json``; a cache-only build that finds no
usable kernel records the region and task it missed. The Auto-Tuner uses
these records to prepare generated kernels once and to check that every
later run loads the same ones. Recording never fails a compilation.

.. autoclass:: GeneratedKernel

.. autofunction:: generated_kernels

.. autofunction:: record_generated_kernel

.. autoclass:: MissingKernel

.. autofunction:: missing_kernels

.. autofunction:: record_missing_kernel

Validating on real inputs
~~~~~~~~~~~~~~~~~~~~~~~~~

Synthetic inputs cannot cover every value a model produces, such as the
``-inf`` entries of masked attention scores, so candidates are also checked on
the inputs that real model data gives each region. By default, that data is
the example inputs given to ``torch_tensorrt.compile``, as one batch, when they
are tensors rather than ``Input`` shape specifications and the model does not
write to its buffers or inputs; if they cannot run the model, compilation logs
a warning and validates on synthetic inputs alone. The tensors they give the
regions are kept within the 1 GiB limit described below: a region whose
tensors would go past it keeps none and logs a warning that gives their size
and the limit; a kernel searched for it is then checked on synthetic inputs
only, and an identical region's kernel is reused for it without a check.
``validation_inputs`` replaces them with your own data: a reusable iterable of
model input batches.
Each batch is a tensor or a tuple of tensors for the positional inputs given
to ``torch_tensorrt.compile``, followed by a dict of the keyword inputs if it
was given ``kwarg_inputs``; a dict alone holds only keyword inputs. A
``DataLoader`` works::

    loader = torch.utils.data.DataLoader(dataset, batch_size=8, drop_last=True)
    generator = KernelGenerator(validation_inputs=loader)

Compilation runs the model, with its own weights, on the first 8 batches, or
fewer once the tensors kept from them reach 1 GiB, and keeps the tensors that
reach each generated region. Every candidate, final validation, and cache
check is then also checked on them, as cases named ``sample 1``, ``sample 2``,
and so on, and the synthetic cases are derived from them, as described under
Acceptance below. Regions are compiled for static shapes, so a batch must
match the compiled input shapes and dtypes (a ``DataLoader`` may need
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
objective in this compilation. It is skipped when the region has no tensors,
when the kernel already passed the same tensors, or when they come from the
example inputs and the kernel's checks already covered each of the region's
inputs: the input's finite values lie within the finite range that the input
had in one set of tensors the kernel passed, or within overlapping such
ranges, but never in a gap between them; the ranges of integer-valued inputs,
such as positions, cover integers only; and an input with at most two
distinct values, such as a 0/1 or an additive mask, must equal one the kernel
passed. These are the ranges of the tensors themselves: the magnitude cases
described under Acceptance do not widen them. Once the kernel passes a
region's check, that region's ranges count as covered too. A region whose
tensors fail it searches again, on its own tensors and those of the identical
regions before it, and every identical region then uses the new kernel.
Failure feedback sent to the author can include values from these tensors.

Runs that search, and refused cache-only runs, keep their ``samples.pt``, and
runs are never pruned. To reclaim space, delete a task directory's ``run-*``
directories other than the one its ``accepted.json`` names (all of them for a
task compiled with ``use_cache=False``), while no compilation from that cache is
running.

Acceptance
~~~~~~~~~~

* ``objective="candidate"`` (default) selects the fastest correct candidate,
  even one slower than the captured PyTorch region, and then logs a warning.
  ``"eager"`` and ``"tensorrt"`` also require ``min_speedup`` over the captured
  PyTorch region or a TensorRT engine built from it with your engine build
  settings, after allowing for measured timing noise.
* Search evaluations screen candidates and tuning variants
  (``SearchPolicy(screening="launch")``, the default): each candidate is
  compiled and registered as the plugin an engine would embed, which applies
  the plugin's own checks, such as the shared memory a TensorRT plugin launch
  allows a Triton kernel, and is then launched directly, which saves a
  TensorRT engine build per evaluation. A Triton candidate that needs more
  shared memory than that is evaluated as the first smaller variant that fits,
  if any, trying one pipeline stage and then halved tile constants, unless
  ``SearchPolicy(auto_fit=False)``. Finalists and cached kernels are
  always checked again as plugins inside a trial engine, and only those engine
  measurements decide ``"eager"`` and ``"tensorrt"``; screened timings only
  rank candidates. A problem that only an engine build reveals therefore
  surfaces in final validation, which tries up to ``finalists`` candidates.
  Under ``"tensorrt"``, screens still build the TensorRT baseline until one
  search evaluation has built it, so a region that TensorRT cannot build stops
  the search after its first evaluation. ``screening="engine"`` builds an
  engine, and for ``"tensorrt"`` a baseline engine, for every evaluation.
* A kernel is also launched directly on tensors inside sentinel padding, which
  rejects writes outside its tensors. Without ``aligned_io=True``, these
  launches, and every screened launch, timed ones included, pass pointers
  aligned only to their element size, as an engine can give a plugin: a kernel
  that assumes more alignment, such as with ``float4`` loads, fails, and
  Triton's and cuTile's own launchers, which compile for the alignment of the
  pointers they are given, run the code that the plugin runs. Trial engines
  keep TensorRT's aligned buffers.
* Without explicit ``rtol``/``atol``, outputs are compared with the exact
  result: the reference evaluated in float64, on the GPU or, when the GPU lacks
  the memory or a float64 kernel, on the CPU, and never with TF32 (a region
  whose inputs and outputs are all float16 or bfloat16 may use float32
  instead). Each element may deviate from it by the output dtype's default
  ``torch.testing.assert_close`` tolerance plus a small multiple of eager
  PyTorch's own error, and the error over each whole output may be at most
  twice eager PyTorch's, or twice a small floor, whichever is larger.
  Reductions and matrix products therefore need not reproduce PyTorch's
  summation order, while elementwise results stay tight. Eager PyTorch's own
  error is measured with TF32 off, for cuDNN convolutions too, although
  PyTorch's defaults let them use it: under the default
  ``fp32_matmul_precision="highest"``, float32 matrix products and
  convolutions are held to IEEE float32 accuracy.
  ``fp32_matmul_precision="high"`` also allows the error that TF32 matrix
  multiplication and convolution give eager PyTorch. A reference that cannot
  be evaluated without rounding fails the evaluation instead of falling back
  to eager PyTorch. With ``rtol`` and ``atol`` set, outputs are compared at
  those tolerances with eager PyTorch's, run at PyTorch's default TF32
  settings, instead.
* Each correctness check runs ``num_tests`` random cases; magnitude cases,
  which scale the inputs by 0.001 and 100 or offset them by +100 and -100;
  constant cases, which set them to 0, -1 and 1; with ``check_nonfinite``,
  cases that inject a NaN or an infinity into each input; and then the
  samples. With samples, every synthetic case starts from one: random cases
  keep continuous values within the sampled range, magnitude and constant
  cases keep each input's sign, and magnitude cases do not scale
  integer-valued inputs, such as positions. The cases scaled by 100 or offset
  then leave the sampled range: they are robustness checks, which require
  finite outputs wherever eager PyTorch's are finite but allow a larger error.
  Without samples, random inputs are standard normal, and if eager PyTorch's
  outputs are mostly NaN or infinite on every random case, the evaluation
  fails and asks for real example tensors or ``validation_inputs``. Wherever
  eager PyTorch's output is NaN or infinite, any candidate value passes. On the
  cases that inject a NaN or an infinity, an output that the injected value
  reaches (one that eager PyTorch makes NaN when the injected element is NaN)
  may also be NaN or infinite where eager PyTorch's is finite, as when ReLU
  computed as ``x * (x > 0)`` turns -inf into NaN instead of 0; a finite value
  there must still be accurate, and outputs that the injected value does not
  reach are checked as on the other cases. The ``instructions`` command below
  prints the exact rules that the author receives.
* Timings are interleaved CUDA graph samples of the isolated region. Eager
  PyTorch is timed at PyTorch's default TF32 settings, under which cuDNN
  convolutions may use TF32 and matmuls do not. Timings do not establish a
  whole-model speedup, and passing checks do not prove correctness for every
  input.

The persistent cache, under ``$XDG_CACHE_HOME/torch_tensorrt/region_kernels`` by
default, keys accepted kernels by the captured computation, tensor contract,
backend toolchain, code-generation options such as ``aligned_io``, instruction
contents, compilation settings, evaluation policy, and ``SearchPolicy``
``tuning`` choices; budgets and the other search settings are not part of the
key. A kernel cached under different tuning choices is therefore not reused,
even if its own values are among the new ones. A cached kernel is
revalidated before reuse, and identical regions in one compilation all use one
kernel. Dry runs (``dryrun=True``) do not invoke the agent; they show the
region executing its PyTorch reference.

Generation requires a POSIX host; static, contiguous, nonempty float16,
bfloat16, or float32 CUDA tensor boundaries; and a TensorRT build with Quick
Deployable Plugin (QDP) support, since the kernel runs in the engine as a
plugin: TensorRT 10.7 or later, other than 10.14.x, and not TensorRT-RTX. Each
region becomes one kernel.
Generated code and extension adapters run locally. Worker subprocesses contain
timeouts and crashes; they are not a sandbox for hostile code. If the compiling
process is killed without a chance to clean up, as by ``SIGKILL`` or a
``SIGTERM`` it does not handle, evaluation workers and provider adapters end
on their own, normally within a second, but an author CLI that is already
running finishes its request first.

A generated region must draw no random numbers. PyTorch tags attention,
dropout and RNN operators as random by name, but attention with
``dropout_p=0``, eval-mode dropout and RNNs draw none and are generated like
other operations: ``F.scaled_dot_product_attention`` reaches the author as
``aten.scaled_dot_product_attention``, or, with ``decompose_attention=True``,
as matrix multiplications and a softmax. Like every boundary of a generated
region, their outputs must be contiguous, and some are not: PyTorch's
memory-efficient attention kernel, which it usually picks for float32 inputs
or an ``attn_mask``, and RNNs with ``batch_first=True`` return non-contiguous
outputs, so such a region is refused before any agent call by an error that
names the operation. Return ``.contiguous()`` of the output inside the region,
which costs nothing where the output is contiguous already;
``decompose_attention=True`` does not change the output's layout, because
compilation captures regions before it decomposes attention. Inputs must be
contiguous too: where a module transposes the query, key and value before
calling attention, call ``.contiguous()`` on them before the region or move the
transposes into it. Calls that draw
random numbers, such as dropout with ``training=True``, and other operators
that PyTorch tags as random, including fused attention operators such as
``aten._scaled_dot_product_flash_attention``, are refused before any agent
call by an error that names the operator.

When generation fails
~~~~~~~~~~~~~~~~~~~~~

Before the first search, compilation checks every generated region for what
generation requires and no agent can change: the requirements above, its
reference graph (ATen operators, none of the random operators refused above,
and tensor constants only as inputs), the backend's toolchain, and whether
generation is allowed (see Cache-only compilation below). A region that fails raises
``RegionError`` naming it, before any agent call. These checks also read the
generator's instruction files and check its ``SearchPolicy`` tuning choices
against the backend.

A search that ends without an accepted kernel also raises ``RegionError``, and
``torch_tensorrt.compile`` fails: nothing falls back to running the region's
PyTorch code or to TensorRT's own conversion of it. A search ends that way when
none of its candidates passes the checks and then meets ``objective`` and
``min_speedup`` in final validation: for example, when every proposal is wrong,
when correct kernels are too slow, or when the provider fails (its CLI is
missing or not logged in, or the model API stays unavailable) or ``timeout_s``
runs out before any kernel passes. The error gives the reason and the run's
artifacts. Kernels already accepted for other regions stay cached (with
``use_cache=True``, the default), so compiling again revalidates them rather
than searching again. To proceed:

* compile again with ``SearchPolicy(resume=True)``, which continues the
  checkpointed search and remeasures its retained candidates, or with a larger
  budget (``max_attempts``, ``timeout_s``, or ``max_candidates``);
* lower ``min_speedup``, or choose ``objective="candidate"``, if correct
  kernels were too slow;
* supply a kernel with ``fused_kernel(kernel=...)``; or
* remove the annotation, so that TensorRT compiles the operations as usual.

A dry run (``dryrun=True``) makes the same checks, logs each failure as a
warning naming the region, and lists the failures in its report. It calls no
agent: an operator that runs the region's PyTorch reference stands in for the
kernel, and partitioning places it as it would place the kernel's plugin, so
the report shows whether the region would run inside a TensorRT engine, as
compiling requires. A region whose boundaries no plugin can have runs its
reference body outside TensorRT instead, and the report says why. A dry run
cannot tell whether a search will find a kernel or whether a cache-only build
has one cached, and it runs neither ``validation_inputs`` nor the example
inputs.

Cache-only compilation
~~~~~~~~~~~~~~~~~~~~~~

CI and production builds can forbid generation, so that compilation never
calls an agent: pass ``allow_generation=False``, or set the environment
variable ``TORCH_TENSORRT_REGION_GENERATION=0`` to apply it to every generator
without code changes (any value other than ``0`` or ``1`` is an error).
Compilation then reuses a cached kernel after revalidating it, and a cache
miss or a failed revalidation raises ``RegionError`` without calling the
provider. Under the environment variable, revalidation checks correctness
only: the kernel's speed was measured when it was accepted, and a busy GPU
cannot fail the build. ``allow_generation=False`` revalidates fully, speed
included (see below). Generate the kernel first by compiling once with generation allowed,
on the same GPU type with the same toolchain and settings,
and give the cache-only build that ``cache_dir``.

The ``cache_dir`` must exist, and it must be owned by and writable by the user
that compiles: every compilation records a run there, even one that reuses a
cached kernel, and a directory owned by another user is refused. When you copy
a cache into an image, ``chown`` it to the user that runs the build. A
cache-only build never creates a missing ``cache_dir``.

Revalidation repeats the final check, so under ``objective="eager"`` or
``"tensorrt"`` it also remeasures the speedup, which a busy GPU can push below
``min_speedup`` (except under ``TORCH_TENSORRT_REGION_GENERATION=0``, which
checks correctness only). The error explains a refusal: after a cache miss it names the
``task.json`` fields in which a kernel cached for the same captured computation
differs, such as ``target`` after a GPU or toolchain change, or
``search.tuning`` after a change of the tuning choices; after a failed
revalidation it gives the stage and error, or the remeasured speedup, and the
run keeps the full report in ``cache_check.json``. A revalidation that could
not complete, at stage ``budget``, ``setup``, ``timeout``, ``baseline``, or
``resources``, does not show that the kernel is wrong: retry it, for example on
a less busy GPU or with larger ``timeout_s`` and ``candidate_timeout_s``,
before generating a new kernel. When evaluating again cannot change the
outcome, as for a reference that cannot be evaluated without rounding, the
error says to resolve the cause it names instead. A kernel refused only for
its remeasured speedup is correct, so the error suggests retrying on a less
busy GPU before searching for a faster one.

Data and trust
~~~~~~~~~~~~~~

* The provider receives the region's captured graph code, its non-tensor
  constants, tensor shapes and dtypes, the instructions, and measured feedback,
  and runs under your CLI login and account. Tensor values are not sent, except
  that failure feedback can quote values from the validation samples, which
  come from ``validation_inputs`` or, by default, from the example inputs given
  to ``torch_tensorrt.compile``. Provider logs, evaluation worker logs and
  reports, and the evidence pages the provider can read have the values of
  environment variables whose names contain ``KEY``, ``TOKEN``, ``SECRET``,
  ``PASS``, ``AUTH`` or ``CREDENTIAL``, and common access-token formats,
  replaced with ``[redacted]``. This catches accidental leaks, not a kernel
  written to disguise a value.
* For the Triton, cuTile and CuTe DSL backends, a provider's reply is Python
  source, and it runs as Python with your privileges, environment and files:
  evaluation workers import and compile each candidate, the compiling process
  loads the accepted kernel, and so does every later build that reuses it from
  the cache. Running the provider CLI without tools does not change this.
  Before a candidate runs, a source check lets it import only its kernel
  language and use only that language's functions, types and constants, its
  own helpers, and the builtins ``abs``, ``bool``, ``float``, ``int``, ``len``,
  ``max``, ``min``, ``range`` and ``tuple``; it rejects code that rebinds these
  names or reaches other modules through them. The check is a guard against
  mistakes, not a sandbox: a reply written to defeat it can run any code as
  you. Use only providers, models and instructions that you trust.
* Runs and the cache are created readable only by you. A cache directory owned
  by another user is refused, and one that others can write is reported,
  because a planted kernel would be revalidated and then loaded.
* Runs store the reference as JSON and validation samples as plain tensors
  that load with ``torch.load(weights_only=True)``, so loading a run executes
  no code from it. The reference can call only ATen and prims operators and
  Python's indexing and arithmetic operators (``getitem``, ``add``, ``sub``,
  ``mul``, ``truediv``, ``floordiv``, ``neg``). Loading refuses operators that
  PyTorch tags as random, except attention, dropout and RNN calls whose
  arguments turn their random numbers off, Inductor's internal primitives,
  operators that access files, and operators that declare they mutate their
  arguments. Kernel source from a run is validated before it is loaded.
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
recorded search policy, budgets, validation and code-generation settings, and
exact instruction snapshot unless options override them, and uses the run's
own cache root unless ``--cache-dir`` names another, so an accepted kernel is
revalidated rather than searched for again; see ``--help``. The repository's
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
``resource``, and ``offset``. A reply that breaks this protocol is not used:
the controller names the defect and asks again, up to ``max_protocol_retries``
times per author round. Raising ``TimeoutError`` ends only the current author
round, unless too little search time remains for another; any other exception
ends the search.

A backend implements the internal ``_generation._backends.GenerationBackend``
protocol: ``name``, ``version``, ``proposal_schema``, ``instruction_resources``,
``target_metadata()``, ``validate_candidate()``, and ``make_registration()``,
which returns a ``KernelSpec`` or a ``functools.partial`` of
``torch_tensorrt.kernels.triton_op``, ``cutile_op``, or ``ptx_op``. An optional
``launch()`` method enables the out-of-bounds write check and launch screening,
which checks and times candidates through it: it must run the kernel its
plugin runs, with the same launch settings, on tensors that are aligned only
to their element size unless the generator sets ``aligned_io``, and be safe to
capture in a CUDA graph, and ``make_registration()`` should reject candidates
that the launch can run but the plugin cannot. Otherwise use
``SearchPolicy(screening="engine")``.
A ``make_registration()`` that declares an ``options`` parameter receives the
generator's code-generation options, currently ``{"aligned_io": bool}``. An
optional ``fit_shared_memory()`` method lets ``SearchPolicy.auto_fit``
evaluate a smaller variant of a candidate whose registration raised
``_generation._backends.SharedMemoryLimitError``.
Instruction resources may use ``package:relative/path.md`` names.
Natural-language instructions alone cannot add a kernel language.

Errors
------

.. autoexception:: RegionError

.. autoexception:: RegionCaptureError
