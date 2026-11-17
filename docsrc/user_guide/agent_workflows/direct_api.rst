Direct Python APIs
==================

Use an existing CUDA-capable Torch-TensorRT installation that contains the
report APIs. Both ``torch_tensorrt.dynamo.analyze()`` and
``torch_tensorrt.dynamo.compile_with_report()`` accept a
``torch.export.ExportedProgram`` plus positional/keyword inputs and supported
Dynamo compilation settings. They copy the exported graph and weights before
lowering, preserving the caller's exported program and Torch random generators.
The report APIs and legacy AOT compiler entrypoints share a process-wide lock
because converter configuration is shared. ``compile()``, ``compile_module()``,
and cross-compilation participate in that serialization. A Debugger context holds
the same lock throughout its lifetime so its temporary configuration stays with
its compilation. Direct concurrent mutation of converter registry configuration
outside these entrypoints is unsupported.

Analysis and compilation
------------------------

This example prepares an FP16 graph explicitly. The modern Dynamo compiler uses
the graph's tensor dtypes; these APIs do not silently cast a model or its inputs.

.. code-block:: python

   import torch
   from torch_tensorrt.dynamo import analyze, compile_with_report
   from torch_tensorrt.reporting import CompilationFailed, ReportStatus

   class Add(torch.nn.Module):
       def forward(self, x):
           return x + x

   reference = Add().cuda().eval()
   x = torch.randn(2, 3, device="cuda", dtype=torch.float16)
   exported = torch.export.export(reference, (x,))

   analysis = analyze(exported, arg_inputs=(x,), min_block_size=1)
   if analysis.status == ReportStatus.FAILED:
       raise RuntimeError(analysis.failure.message)
   print(analysis.to_json())

   try:
       result = compile_with_report(
           exported, arg_inputs=(x,), min_block_size=1,
           diagnostics_dir="compiler-diagnostics",
       )
   except CompilationFailed as error:
       print(error.report.to_json())
       raise
   compiled = result.module
   print(result.report.to_json())

``analyze()`` returns an ``AnalysisReport``. Its status is ``SUCCEEDED``
when analysis completes, or ``FAILED``/``CANCELLED`` with structured failure
details otherwise.
``compile_with_report()`` returns ``CompilationResult(module, report)`` on
success and raises ``CompilationFailed`` with its failed report on an error.
Neither changes the return type of the existing compilation entrypoints.
Unknown report-API settings are rejected; the current report API does not
execute autocast calibration dataloaders or enable the autocast calibration
workflow.

Reports count computational nodes in the lowered graph. The coverage ratio is
eligible computational nodes divided by all computational nodes, with zero for
an empty denominator. Eligibility and final assignment differ when a partition
is too small. Structural operations, state nodes, exclusions, unsupported
operations, and below-minimum assignments are reported separately. Consult
``checks`` and node ``confidence`` when deciding what a result establishes.

Events and diagnostic files
---------------------------

Pass ``event_sink=callback`` to either report API. The callback receives typed
``Event`` objects containing a job ID, stage, sequence number, UTC timestamp,
and measured details. ``event.to_json()`` is suitable for a JSON-lines file.
Optional callback exceptions are retained as diagnostics and do not fail an
otherwise successful compilation. A sink can set ``critical = True`` to make
event-persistence failures abort the operation.

``diagnostics_dir`` on ``compile_with_report()`` captures the exported graph
and compiler logs. Keep full diagnostics local; use bounded report fields for
assistant summaries. Additional engine-layer diagnostics depend on the
compiler/debug facilities enabled by the caller.

Validate and measure compiled modules
-------------------------------------

``InputSample`` carries an argument tuple, keyword arguments, an optional profile
label, sample ID, and display label. Input tensors are already on their intended
device; the verification APIs do not perform implicit device or dtype casts.

.. code-block:: python

   from torch_tensorrt.verification import InputSample, benchmark, validate

   sample = InputSample(args=(x,), profile="static", sample_id="sample-0")

   validation = validate(
       reference, compiled, [sample], atol=1e-3, rtol=1e-3,
       relative_floor=1e-12, seed=0,
   )
   if not validation.passed:
       raise RuntimeError(validation.failure.message)

   timing = benchmark(
       reference, compiled, sample, warmup=10, iterations=100,
       percentiles=(50, 90, 99), seed=0,
   )
   print(validation.to_json())
   print(timing.to_json())

The direct APIs return structured reports but do not choose a persistence
policy. Callers that save and reload a compiled module should validate the
reloaded module at every intended dynamic-profile point.

Verification copies the two modules, uses eval/inference mode, and preserves
caller inputs, tensor aliases, model state, and Torch/Python/loaded NumPy RNG
state. Exported graph modules preserve their frozen capture semantics and shape
guards. Input or registered-state mutation is rejected with ``VerificationError``;
model copying, cross-aliasing between inputs and module state, unsupported tensor
layouts, and multi-CUDA-device execution are also outside the current contract.
Evaluation should be captured before exporting a reference model.

See :doc:`contracts_testing` for the exact numerical, timing, and hash rules.
