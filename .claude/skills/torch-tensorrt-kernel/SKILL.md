---
name: torch-tensorrt-kernel
description: Generate, inspect, or evaluate agent-authored Torch-TensorRT region kernels using the package's shared instructions and recorded task artifacts.
---

# Torch-TensorRT region kernels

Use the package's instruction library and generation controller so interactive
work follows the same backend contract as compilation. Keep the user's selected
provider and backend; Claude Code and the kernel language are independent.

Read the canonical guidance for the selected backend:

```bash
python -m torch_tensorrt.dynamo.region_compilation._generation instructions --backend triton
```

Replace `triton` with the requested registered backend. This command reads the
resources shipped with the installed package and ends with the backend's
proposal schema, which lists the exact fields of a candidate JSON object. Do
not maintain another copy of the authoring instructions in this skill.

For an existing task or run, inspect its recorded evidence. PATH may be a run
directory, a file inside one, or the task cache directory that a generation
error reports as `Artifacts:`:

```bash
python -m torch_tensorrt.dynamo.region_compilation._generation inspect PATH
```

When asked to generate candidates, use the controller with the saved task and
the selected provider/backend. Replay restores the task's search policy, budget,
validation, code-generation, and compile options; flags override them. It uses
the run's own cache root unless `--cache-dir` names another. When the task already
has an accepted kernel in the cache, the controller revalidates it instead of
searching and the output reports `"cache_hit": true`; pass `--no-cache` to
search again:

```bash
python -m torch_tensorrt.dynamo.region_compilation._generation generate PATH --provider claude --backend triton
```

When asked to evaluate a candidate, use the existing worker. It checks the
candidate on fresh holdout inputs with the task's settings and exits nonzero
unless the candidate meets the task's objective:

```bash
python -m torch_tensorrt.dynamo.region_compilation._generation evaluate PATH --candidate CANDIDATE_PATH
```

Use command help for the accepted artifact paths and options. Pass
`--output FILE` before the command to also write its JSON result to a file;
TensorRT may print warnings to stdout when the package is imported. Report compiler,
correctness, and timing outcomes from saved results. Distinguish proposal
changes from measured gains, and region timing from whole-model performance.
Do not replace the evaluation path with agent-written correctness or timing
claims. Instructions and artifact inspection do not require calling a model.
