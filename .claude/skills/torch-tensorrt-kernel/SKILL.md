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
resources shipped with the installed package; do not maintain another copy of
the authoring instructions in this skill.

For an existing task or run, inspect its recorded evidence:

```bash
python -m torch_tensorrt.dynamo.region_compilation._generation inspect PATH
```

When asked to generate candidates, use the controller with the saved task and
the selected provider/backend:

```bash
python -m torch_tensorrt.dynamo.region_compilation._generation generate PATH --provider claude --backend triton
```

When asked to evaluate a candidate, use the existing worker:

```bash
python -m torch_tensorrt.dynamo.region_compilation._generation evaluate PATH --candidate CANDIDATE_PATH --timeout 120
```

Use command help for the accepted artifact paths and options. Report compiler,
correctness, and timing outcomes from saved results. Distinguish proposal
changes from measured gains, and region timing from whole-model performance.
Do not replace the evaluation path with agent-written correctness or timing
claims. Instructions and artifact inspection do not require calling a model.
