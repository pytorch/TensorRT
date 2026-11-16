Evidence contracts and testing
==============================

All core evidence contracts currently use the experimental schema version
``0.1`` and explicit JSON serialization. Reports use UTC timestamps and enum
status/stage/category values. Unknown versions, fields, invalid types, duplicate
JSON keys, and nonfinite JSON numbers are rejected. Object keys serialize
deterministically; timestamps and measured results naturally differ between
runs. Published schemas live under ``schemas/reporting/v0.1``; example
documents used by tests live under ``tests/py/reporting/golden/v0.1``. The
``0.1`` contract is experimental, but published schema files are immutable.
Incompatible changes must introduce a new schema version and regenerate its
schemas and golden documents. After ``1.0``, breaking changes require a new
major schema version.

Numerical semantics
-------------------

Validation requires identical pytree structure, leaf types, tensor shapes, and
tensor dtypes. Dictionary insertion order and list/tuple distinctions are part
of the strict structure. Integer and boolean leaves use exact equality.
Floating/complex tensor calculations use FP64/complex128 comparison arithmetic,
preserving precision that would be lost by an unconditional FP32 cast.

The pass rule is ``abs(actual-reference) <= atol + rtol*abs(reference)``.
Absolute/relative tolerances must be finite and nonnegative. Relative error
statistics use ``abs(actual-reference) / max(abs(reference), relative_floor)``;
the floor is finite, strictly positive, and recorded. This diagnostic statistic
does not replace the pass rule near zero. Complex errors use magnitude.

NaN or infinity in either output always fails, including matching nonfinite
values. Reports count those values separately for reference and compiled
outputs. Empty tensors pass after shape/dtype/layout agreement and have absent
error statistics. Unrepresentable statistics are ``null``, never JSON NaN/Inf.
Finite extreme values are compared without allowing arithmetic overflow to make
``inf <= inf`` pass. A failed comparison produces a failed validation report
with a categorized failure and a path to the output leaf.

Each validation sample uses ``(seed + sample_index) % 2**64`` for both models;
tested profile labels and input fingerprints are retained. The fingerprints
cover call structure, tensor metadata, exact logical values, and aliasing.
Evidence applies only
to those values and shape points. It does not certify all values within a dynamic
profile or promise deterministic TensorRT tactics across builds.

Timing semantics
----------------

Benchmarking measures serial forward invocations on one prepared input sample
under inference mode. Each call resets the declared seed and copies inputs while
preserving tensor storage aliases, views, and strides. Models use isolated eval
copies; exported modules retain their captured graph semantics. Mutations of
inputs or registered model state are rejected. Mutation checks do not cover
ordinary Python attributes or unregistered tensor attributes; models relying on
such mutable state are outside the verification protocol.

CPU measurements use wall-clock time. CUDA measurements report separate current
stream event durations and synchronized wall-clock durations. CUDA work on other
streams must join the measured stream before forward returns. The measurement
device is inferred from inputs and registered parameters/buffers; forward calls
that create work on another device are outside the protocol. Warmup, cloning,
state checks, model copying, preprocessing, and input transfers are outside the
timing scope. Serialized invocation and synchronization are recorded; this is
not a concurrent throughput saturation test.

Reports retain every measured sample and median/mean/min/max statistics.
Percentiles interpolate linearly between sorted samples at position
``p/100*(n-1)``. Speedup is the reference median divided by the compiled median
for the same timing scope. A zero compiled CUDA median does not produce a device
speedup. Serial invocation throughput is ``1000 / median_wall_ms``; an explicit
batch size scales this to items per second. GPU-isolation status is caller
supplied and defaults to ``unknown``. CUDA graph benchmark modes are outside
the current v0.1 contract.

Report persistence
------------------

The Core APIs return typed reports and canonical JSON but do not choose an
artifact directory, retention period, or publication policy. The
``ArtifactManifest`` and ``EnvironmentReport`` contracts define interoperable
document shapes for persistence layers without requiring one in Core.
Integrity hashes are evidence against a trusted manifest, not signatures proving
authenticity. A persistence layer must finish writing files before hashing and
must define its own atomicity, immutability, path, and symlink policies.

CPU and GPU checks
------------------

The hosted CPU CI workflow runs contract/schema/golden tests, regression tests
for the published-schema compatibility gate, standalone observer-context tests,
and isolated validation/benchmark tests. It installs CPU Torch and test
requirements without installing TensorRT or native Torch-TensorRT. The tests load
the new Python modules with isolated loaders so collecting them does not run the
native package initializer.

Run the CPU groups from a checkout with CPU Torch, pytest, jsonschema, and NumPy
installed:

.. code-block:: console

   python -m pytest tests/py/reporting tests/py/verification/test_verification.py -q -o addopts="" --noconftest
   python -m pytest tests/py/dynamo/test_observer_contexts.py tools/test_check_agent_contracts.py -q -o addopts="" --noconftest

GPU checks belong in an already provisioned GPU test lane. They compile,
validate, and measure a small FP16 model, and check structured compiler reports.
The CUDA integration test skips when CUDA is unavailable; a skip is not evidence
that the engine workflow passed. Run it explicitly in that environment:

.. code-block:: console

   python -m pytest tests/py/verification/test_verification_gpu.py tests/py/dynamo/runtime/test_agent_reports.py -q -o addopts=""

Latency assertions check valid samples and internal speedup calculations instead
of fixed performance thresholds. CPU CI does not provision a GPU, verify native
engine builds, or measure GPU speedups. The workflow does not create a new
self-hosted runner deployment.
