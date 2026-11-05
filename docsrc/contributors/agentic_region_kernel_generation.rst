.. _agentic_region_kernel_generation_design:

Agentic Region Kernel Generation
================================

.. note::

   This page records the architecture and design goals of compile-time kernel
   generation for ``fused_kernel`` regions. The
   :ref:`region API documentation <torch_tensorrt_region_py>` describes
   configuration, replay commands, and extension contracts, and the
   :ref:`runnable example <agentic_region_kernel_generation_example>` covers
   prerequisites and verifies execution inside the compiled engine.

Users mark a functional region with ``torch_tensorrt.region.fused_kernel`` and a
``KernelGenerator``. During compilation, an agent proposes GPU kernels and
Torch-TensorRT compiles, checks, and measures them. The accepted kernel becomes a
Quick Deployable Plugin (QDP) in the final TensorRT engine. Inference runs the
compiled engine without an agent.

Design goals
------------

* Keep authoring, instructions, and evaluation separate, while using
  Torch-TensorRT's own compiler and evaluator.
* Let users choose the model provider independently from the kernel language.
  Built-in providers use logged-in Claude or Codex CLIs; built-in backends are
  Triton, NVIDIA cuTile, and CUDA C++ compiled through CUDA Python and NVRTC.
* Let the controller decide acceptance from measured correctness and
  performance. The author can change source and launch settings, but cannot
  replace the evaluator or supply its own acceptance result.
* Make searches bounded, inspectable, and reusable. Save the task, exact
  instructions, proposals, and measured feedback; revalidate a cached winner
  before using it in another compilation.

Component responsibilities
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Responsibility
   * - Region capture
     - Preserve the selected computation and its static tensor contract, and
       record real region inputs from validation batches
   * - Generation controller
     - Own budgets, evidence access, tuning, ranking, resume, and final selection
   * - Provider adapter
     - Return proposals or evidence requests under the supplied response schema
   * - Backend adapter
     - Define candidate syntax, validate launch settings, and compile and register kernels
   * - Evaluation worker
     - Check outputs and input immutability and measure execution, by direct
       launches during the search and in a trial plugin engine for finalists
   * - Existing plugin pipeline
     - Register the accepted kernel and compile the surrounding TensorRT engine

Provider and backend extensions use explicit, importable ``module:attribute``
adapters. Instructions alone cannot add a compiler or registration path.

.. code-block:: text

    Capture region, snapshot instructions
        │
        ▼
    Cached candidate? ──── fresh validation passes ───────────────┐
        │ miss, or validation fails (error if generation is off)  │
        ▼                                                         │
    Author returns candidates or requests evidence ◄────┐         │
        │                                               │         │
        ▼                                               │ feedback│
    Controller expands tuning variants                  │ and     │
        │                                               │ budget  │
        ▼                                               │ remain  │
    Worker launches, checks, and measures ──────────────┘         │
        │ budget exhausted, policy stop, or author stop           │
        ▼                                                         │
    Fresh final validation of finalists in an engine              │
        │ passes the objective                                    │
        ▼                                                         ▼
    Register the kernel plugin ──────────────► Build the final TensorRT engine

Search and author protocol
--------------------------

``max_attempts`` bounds *author rounds*. A round can return up to
``SearchPolicy.batch_size`` candidates, and the controller can expand each into
bounded launch variants from explicit tuning domains. ``max_candidates`` limits
actual search evaluations, including tuning, resumed candidates, and requested
diagnostics; by default it allows ``max_attempts`` rounds of ``batch_size``
proposals, each with its tuning variants. Cache revalidation and
final validation do not consume candidate slots. Final validation reserves time
within the overall search timeout, and ``finalists`` bounds the ranked
alternatives that can receive that check.

Each author round shares its ``author_timeout_s`` allowance across proposal calls
and at most ``max_context_requests`` evidence reads or diagnostic requests.
``candidate_timeout_s`` bounds each worker evaluation; ``timeout_s`` bounds the
search. Resuming restores compatible evidence and remeasures retained candidates
with fresh invocation budgets.

The response envelope contains ``action``, ``candidates``, ``resource``, and
``offset``; unused fields are null. Candidate payloads follow the selected
backend's schema, and custom providers return the same envelope.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Action
     - Meaning
   * - ``propose``
     - Supply a bounded ``candidates`` batch (one or more proposals)
   * - ``read``
     - Request a page of a named evidence resource at ``offset``
   * - ``diagnose``
     - Request a measured diagnostic of the incumbent when diagnostics are enabled
   * - ``stop``
     - End authoring; the controller still validates any selected finalist

Evidence consists of immutable snapshots registered by the controller. Reads
return bounded pages and continuation offsets, and cannot address arbitrary
paths. Diagnostics share both context and evaluation budgets.

Instructions, acceptance, and replay
------------------------------------

The packaged author skill and backend references are loaded with
``importlib.resources``, so installed wheels work without repository paths.
Inline guidance and explicit instruction files are snapshotted with the packaged
resources. The repository's interactive Claude Code skill
(``.claude/skills/torch-tensorrt-kernel/SKILL.md``) uses the same instructions
and controller.

Candidates share search validation cases. Finalists and cached candidates
receive fresh validation. Unless explicit tolerances are configured, outputs
are compared with a float64 evaluation of the reference, allowing a small
multiple of eager PyTorch's own rounding error, so reductions and contractions
are not held to a bit-exact summation order. A guard-band launch places every
tensor inside sentinel padding and rejects kernels that write outside their
tensors or modify inputs, which the trial engine alone cannot observe;
out-of-bounds reads are caught only when the sentinel values reach an output.

Search evaluations screen candidates by default. The worker still compiles
and registers each candidate as the QDP plugin an engine would embed, so the
plugin path's own checks apply: Triton's and cuTile's ahead-of-time
compilation and launch metadata checks, and, for Triton, the shared memory
limit of a plugin launch, which Triton's own launcher would lift by opting in
to more. It then launches the kernel directly through the backend's
``launch`` on the shared cases and times those launches in CUDA graphs,
without building a TensorRT engine. Finalists, cache revalidation, and
replayed evaluations always rebuild the kernel as a plugin in a trial engine
and repeat every check there. A screened report records ``execution: launch``
and leaves ``meets_objective`` undecided under the ``eager`` and ``tensorrt``
objectives: screened candidates stay eligible as finalists, ranked by their
screened latency, and only final engine measurements decide the speedup.
Under ``tensorrt``, screens still build the baseline engine until one search
evaluation gets past it, because no proposal can fix a baseline failure and
it ends the search. Problems that only an engine build reveals surface in
final validation, where ``finalists`` bounds the alternatives tried; a
screened finalist that fails there to compile or to build an engine also
leaves the resume checkpoint, since that failure reproduces.
``SearchPolicy(screening="engine")`` restores an engine, and for the
``tensorrt`` objective a baseline engine, for every evaluation.

Synthetic cases can miss values that real models produce. With
``validation_inputs``, ``materialize_regions`` first runs the parent graph on
up to 8 user batches with an FX interpreter, after flattening each batch by
the exported input spec. No region has been replaced yet, so each region still
executes its PyTorch reference with the model's weights, and the interpreter
copies the tensors at every generated region's boundary to the CPU. Parameter
and buffer operands are copied once for all batches, and capture stops early
once the copies reach ``MAX_SAMPLE_BYTES`` (1 GiB). The controller checks these
samples against the static region contract before any author call, saves them
with the run as ``samples.pt`` (a list of per-batch tensor lists that loads
with ``weights_only=True``), and records their digest in the task. Every
evaluation appends them to its synthetic cases, and CLI replay and
re-evaluation use a run's samples only when they match that digest. A run
that reuses a kernel deletes its copy after the check. Runs that search, and
refused cache-only runs, keep theirs as artifacts, and old runs are not
pruned: to reclaim space, delete a task's ``run-*`` directories other than the
one its ``accepted.json`` names (all of them for ``use_cache=False``), once no
compilation from that cache is running, because a cache hit registers its
kernel from its own run and cuTile rereads kernel source from it. The captured
tensors stay in host memory until every region is generated, so that a later
identical region's search can also check them, within the same budget; they
are released before TensorRT engines build.

The ``candidate`` objective ranks correct generated kernels; ``eager`` and
``tensorrt`` also require the configured speedup against the corresponding
region baseline, built with the user's relevant compilation settings.
Interleaved measurements and a median absolute deviation allowance reduce
sensitivity to timing noise. These checks do not prove correctness for every
input or establish whole-model speedup.

Cache identity binds the captured computation, tensor contract, backend and
toolchain, exact instruction contents, compilation settings, objective, and
evaluation policy; it does not depend on file locations, device indices, or
validation samples, which change with the weights and upstream numerics: a
cached kernel is reused only after a holdout check on the current samples.
With generation disabled
(``allow_generation=False`` or ``TORCH_TENSORRT_REGION_GENERATION=0``), a
cache miss or failed revalidation ends compilation with a ``RegionError``
before any provider call, so CI and production builds rely on cached kernels
alone. The error names the task fields in which a kernel cached for the same
computation differs, or the failed revalidation's stage and error or
remeasured speedup. It advises a retry before a new search when the check
could not complete or found a correct kernel too slow. A missing cache
directory is reported rather than created.
Identical regions in one compilation share one generated kernel through a
memo keyed by the task. A region whose samples that kernel has not passed yet
reuses it after a holdout check on them, also with ``use_cache=False``. That
check covers correctness only, since the kernel already met the objective in
this compilation, and an inconclusive one is retried once. If it fails, the
region's search also validates on the samples of the identical regions before
it, so the kernel it publishes passes them all. ``materialize_regions`` then
registers that final kernel for every identical region generated for the same
task, including those before it, without starting another search. Dry runs
show the PyTorch reference instead of invoking an agent. Saved runs include
the reference, task, instruction bundle, per-attempt feedback, and
search history. Replay with the same backend and no guidance override uses the
exact saved instruction bundle; a missing or mismatched snapshot fails before
execution. Explicit backend or guidance changes start a new bundle.

Current boundaries are static, contiguous, nonempty CUDA float32, float16, or
bfloat16 tensors. Each candidate contains one kernel, and tensor constants must
be boundary inputs. Generated code and extension adapters execute locally;
subprocesses provide timeout and failure containment, not a hostile-code
sandbox.
