# Use evaluator evidence to improve the next proposal

The controller evaluates proposals through TensorRT plugin registration and a
trial engine. Successful source parsing does not mean that compilation,
correctness, or measurement succeeded. The worker's reported stage and outcome
determine which issue to address.

- Source or schema failure: repair the exact backend contract violation.
- Compilation failure: use the actual compiler diagnostic and installed
  toolchain version. Do not guess that an unsupported API exists.
- Correctness failure: inspect the failing case and output; correct indexing,
  broadcasting, dtype transitions, or arithmetic before tuning performance.
- Measurement failure: distinguish unavailable timing from a slow kernel.
- Passing candidate: compare its reported latency with other passing candidates
  under the same measurement conditions. A timing difference within observed
  noise is uncertain.

Use the most recent failures and the incumbent's source and launch settings.
Change source or configuration for a new experiment; unchanged passing candidates reuse their prior evaluation.
The controller retains the best passing candidate if a later experiment fails.

Follow the task's objective. `candidate` ranks correct generated proposals;
`eager` and `tensorrt` require the configured `min_speedup` against the named
PyTorch or fully compiled TensorRT baseline. The ratio is baseline latency
divided by candidate latency. Acceptance allows for observed timing variation:
`baseline_median - baseline_MAD >= min_speedup * (candidate_median + candidate_MAD)`.
MAD is median absolute deviation; this is a noise allowance, not a statistical
confidence guarantee. The controller makes the decision from actual samples.
Selection is followed by a mandatory evaluation on fresh
holdout inputs; passing search cases alone does not finish validation.

Eager reference timing and candidate timing describe the recorded workloads.
They do not establish a whole-model speedup or prove a win over native TensorRT.
Do not invent sample counts, timings, device conditions, or claims of measured
improvement. The remaining wall-time and attempt budgets include generation
and evaluation; choose a useful next experiment within that allowance.
