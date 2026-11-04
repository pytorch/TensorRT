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
     - Preserve the selected computation and its static tensor contract
   * - Generation controller
     - Own budgets, evidence access, tuning, ranking, resume, and final selection
   * - Provider adapter
     - Return proposals or evidence requests under the supplied response schema
   * - Backend adapter
     - Define candidate syntax, validate launch settings, and compile and register kernels
   * - Evaluation worker
     - Build a trial plugin engine, check outputs and input immutability, and measure execution
   * - Existing plugin pipeline
     - Register the accepted kernel and compile the surrounding TensorRT engine

Provider and backend extensions use explicit, importable ``module:attribute``
adapters. Instructions alone cannot add a compiler or registration path.

.. code-block:: text

    Capture region, snapshot instructions
        │
        ▼
    Cached candidate? ──── fresh validation passes ───────────────┐
        │ miss, or validation fails                               │
        ▼                                                         │
    Author returns candidates or requests evidence ◄────┐         │
        │                                               │         │
        ▼                                               │ feedback│
    Controller expands tuning variants                  │ and     │
        │                                               │ budget  │
        ▼                                               │ remain  │
    Worker compiles, checks, and measures ──────────────┘         │
        │ budget exhausted, policy stop, or author stop           │
        ▼                                                         │
    Fresh final validation of ranked finalists                    │
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

The ``candidate`` objective ranks correct generated kernels; ``eager`` and
``tensorrt`` also require the configured speedup against the corresponding
region baseline, built with the user's relevant compilation settings.
Interleaved measurements and a median absolute deviation allowance reduce
sensitivity to timing noise. These checks do not prove correctness for every
input or establish whole-model speedup.

Cache identity binds the captured computation, tensor contract, backend and
toolchain, exact instruction contents, compilation settings, objective, and
evaluation policy; it does not depend on file locations or device indices.
Identical regions in one compilation share one generated kernel, and dry runs
show the PyTorch reference instead of invoking an agent. Saved runs
include the reference, task, instruction bundle, per-attempt feedback, and
search history. Replay with the same backend and no guidance override uses the
exact saved instruction bundle; a missing or mismatched snapshot fails before
execution. Explicit backend or guidance changes start a new bundle.

Current boundaries are static, contiguous, nonempty CUDA float32, float16, or
bfloat16 tensors. Each candidate contains one kernel, and tensor constants must
be boundary inputs. Generated code and extension adapters execute locally;
subprocesses provide timeout and failure containment, not a hostile-code
sandbox.
