# Use evaluator evidence to improve the next proposal

The controller evaluates proposals through TensorRT plugin registration and a
trial engine. Successful source parsing does not mean that compilation,
correctness, or measurement succeeded. A failed report names the stage to
address:

- `source`: the proposal violates the backend contract or schema; repair that
  exact violation.
- `compile` or `engine`: use the actual compiler or TensorRT diagnostic and the
  installed toolchain version. Do not guess that an unsupported API exists.
- `correctness`: inspect the failing case and output; correct indexing,
  masking, broadcasting, dtype transitions, or arithmetic before tuning
  performance. Writes outside a tensor fail here too.
- `benchmark`: the candidate is correct but could not be timed; distinguish
  unavailable timing from a slow kernel.
- `timeout`: the kernel, or building and validating it, exceeded the
  per-candidate time.
- `crash`: native code killed the evaluator, typically through an invalid
  memory access; check indexing, masking, and launch dimensions.
- `author`: the previous author turn produced no usable reply: it broke the
  response protocol or ran out of time. Read the error.
- `setup`, `baseline`, and `budget` concern the controller, the TensorRT
  baseline, or the remaining search time, not the candidate's code. Do not
  rewrite a kernel because of them.

Compare a passing candidate's latency with other passing candidates measured
under the same conditions; a difference within observed noise is uncertain. A
passing report can carry `incumbent_error` when the incumbent could not be
built or timed for comparison; the candidate's own result still stands.

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
