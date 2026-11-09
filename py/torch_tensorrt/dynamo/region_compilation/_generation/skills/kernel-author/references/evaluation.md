# Use evaluator evidence to improve the next proposal

The controller registers each proposal with its backend. During the search it
usually screens the proposal by launching the kernel directly
(`execution: launch`); final validation runs the selected kernel as a TensorRT
plugin inside a trial engine (`execution: engine`). Successful source parsing
does not mean that compilation, correctness, or measurement succeeded. A failed
report names the stage to address:

- `source`: the proposal violates the backend contract or schema; repair that
  exact violation. The history shows the rejected proposal as
  `rejected_proposal`.
- `compile` or `engine`: use the actual compiler or TensorRT diagnostic and the
  installed toolchain version. Do not guess that an unsupported API exists.
  A kernel needing more shared memory than a TensorRT plugin launch provides
  fails here, and what counts depends on the backend. For Triton, the shared
  memory its launch requests must not exceed
  `target.plugin_max_shared_memory_bytes`, and the report names its largest
  shared-memory buffers; for CUDA C++ and CuTe DSL, its static shared memory
  and `shared_mem_bytes` together must not. cuTile requests none at launch:
  its compiler allocates shared memory statically, which a plugin can hold up
  to the GPU's per-block maximum on compute capability 10.0 and newer, 12.0
  included; on 8.x, a kernel whose PTX the CUDA driver cannot load fails
  here, so use smaller tiles. When the evaluator could fit a smaller variant,
  the report carries `fit_log` and `fitted_from`, the candidate shown is the
  variant it evaluated, and later rounds should start from that variant.
- `resources`: compiling, registering, launching or running the candidate
  ran out of device memory, including its runs during the correctness checks.
  Reduce what the kernel allocates or spills, such as large register tiles;
  on a busy GPU the same candidate may pass later.
- `correctness`: inspect the failing case and output; correct indexing,
  masking, broadcasting, dtype transitions, or arithmetic before tuning
  performance. Writes outside a tensor fail here too. Cases named `sample N`
  are region inputs captured from the user's real model inputs; `random N`,
  `scaled`, `offset` and `constant` cases derive from them when they exist
  (only `random 1` keeps the samples' bounds, masks and scalars), and
  `input I[0] = V` cases inject a NaN or an infinity into input I. A
  "robustness case" (`scaled x100`, `offset +100` or `offset -100` with
  samples) leaves the sampled range. It fails on a NaN or infinity where
  eager PyTorch is finite, on an error above 1% of the exact output's norm
  (or 2 times eager's, whichever is larger), or on an element off by more
  than 10% of the output's largest exact value, not on the in-range bounds.
  The message names the worst element and its allowance, or, when every
  element passes, the error over the whole output: a small error shared by
  many elements, such as a wrong scale, decay or window. "Every output is
  finite" means an injected NaN never reached the output and the outputs it
  should reach did not change with the injected element either: the kernel
  ignores that input or hard-codes its value. The message also says where the
  exact reference ran and how far eager PyTorch itself is from it, so compare
  the candidate's error with eager's.
- `benchmark`: the candidate is correct but could not be timed; distinguish
  unavailable timing from a slow kernel.
- `timeout`: the kernel, or building and validating it, exceeded the
  per-candidate time. The limit includes computing every case's expected
  outputs; when the GPU is busy, the float64 reference may run on the CPU and
  take most of it.
- `crash`: native code killed the evaluator, typically through an invalid
  memory access; check indexing, masking, and launch dimensions.
- `author`: the previous author turn produced no usable reply: it broke the
  response protocol again after being asked to repair it, or ran out of time.
  Read the error; `rejected_reply` holds the reply that was not used. A
  proposing reply must put its proposals in `candidates`, never null. A turn
  that the remaining budget shortened and that ran out of time ends the
  search (`budget`) unless another full turn and the time to evaluate it
  still fit.
- `setup`, `baseline`, `budget` and `provider` concern the controller, the
  TensorRT baseline, the remaining search time or the model API, not the
  candidate's code, as when the reference cannot be evaluated without
  rounding, when standard normal inputs leave most outputs NaN or infinite,
  or when the model API stays unavailable after waiting and asking again. Do
  not rewrite a kernel because of them. A setup failure is evaluated once more
  before it is reported, unless evaluating again cannot change it
  (`retryable: false`).

Compare a passing candidate's latency with other passing candidates measured
under the same conditions; a difference within observed noise is uncertain. A
passing report can carry `incumbent_error` when the incumbent could not be
built or timed for comparison; the candidate's own result still stands. It
also records how outputs were compared (`comparison`, `oracle`) and whether
the inputs were `sample-derived` or `unconstrained N(0,1)` (`inputs`).
`inconclusive_cases` lists cases where eager PyTorch's output is NaN or
infinite at more than half of an output's elements, so they checked little
there; `robustness_cases` lists the cases judged as robustness checks.

Use the most recent failures and the incumbent's source and launch settings.
Change source or configuration for a new experiment: an unchanged candidate
reuses its earlier pass or its source, compile, engine, or correctness failure.
The controller retains the best passing candidate if a later experiment fails.

Each prompt states the round's search step, which the controller chooses from
measured outcomes, and the evaluations that remain. Candidate evaluations,
including tuning variants, resumed candidates, and diagnostic evaluations,
consume that allowance. Each candidate in a batch has its own correctness and
timing result; a passing neighbor does not validate the rest of the batch. A
no-improvement limit may stop exploration before the author-round or time
limits are exhausted.

Optional diagnostic feedback contains actual compiler or execution facts. A
missing fact is unavailable, not zero. Only when the task has `diagnostics=true`,
request `diagnose` on
the incumbent when the result would help choose a different experiment. It
shares the bounded context request budget with evidence reads.

Follow the task's objective. `candidate` ranks correct generated proposals;
`eager` and `tensorrt` require the configured `min_speedup` against the named
PyTorch or fully compiled TensorRT baseline. The ratio is baseline latency
divided by candidate latency. Acceptance allows for observed timing variation:
`baseline_median - baseline_MAD >= min_speedup * (candidate_median + candidate_MAD)`.
MAD is median absolute deviation; this is a noise allowance, not a statistical
confidence guarantee. The controller makes the decision from actual samples.
Only engine measurements decide these objectives: a screened report leaves
`meets_objective` null, and its latency ranks it against other screened
candidates.
Selection is followed by mandatory evaluation on fresh holdout inputs; passing
search cases alone does not finish validation. With multiple finalists, a failed
final check can fall through to the next shortlisted candidate. Resumed search
evidence is historical context; retained candidates are measured again before
selection under the new invocation's budgets.

Eager reference timing and candidate timing describe the recorded workloads.
They do not establish a whole-model speedup or prove a win over native TensorRT.
Do not invent sample counts, timings, device conditions, or claims of measured
improvement. The remaining wall-time and attempt budgets include generation
and evaluation; choose a useful next experiment within that allowance.
