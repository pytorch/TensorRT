.. _edge_executorch_integration:

Edge-LLM ExecuTorch Integration
===============================

Status
------

**Proposed design.** This document defines an incremental path from the
experimental Edge exporter to a libtorch-free ExecuTorch program. PI0.5 is the
first target. GR00T and other VLA families should reuse the same contracts after
the PI0.5 path is validated.

The design has two parallel execution tracks:

* a vanilla track that packages component engines with the existing
  ``TensorRTBackend``; and
* a performance track that adds an out-of-tree ``EdgeLLMBackend`` whose
  component handles own the corresponding Edge-LLM runners and state.

The Edge-LLM performance track is the primary integration requested here. The
vanilla track is the bring-up baseline and fallback, not a substitute for the
specialized runtime.

Summary
-------

The integration combines three systems that are currently independent:

* ``tools/hf/exporters`` decomposes a policy and builds TensorRT engines. Its
  outer ``ExportedProgram`` calls path-based ``edge_llm::*`` custom operators.
* ``torch_tensorrt.executorch`` lowers ``tensorrt::execute_engine`` nodes into
  libtorch-free ``TensorRTBackend`` delegates and embeds each engine in a
  ``.pte`` file.
* TensorRT-Edge-LLM supplies TensorRT plugins, reference engine/runtime
  behavior, and a stateful C++ VLA runtime. It does not currently implement an
  ExecuTorch backend.

The performance path replaces the generic path-based engine calls in the outer
Edge graph with named component operators such as
``edge_llm::vision_tower``, ``edge_llm::language_runtime``, and
``edge_llm::action_runtime``. ``EdgeLLMPartitioner`` claims those operators and
packs each component engine, runtime kind, and required metadata for
``EdgeLLMBackend``. On the C++ side, each delegate handle creates the
corresponding Edge-LLM runner, owns its persistent state, and calls the runner
with ExecuTorch tensor arguments.

The existing ``TensorRTPartitioner`` remains responsible for ordinary
``tensorrt::execute_engine`` nodes. A CUDA partitioner may claim supported
residual graph operations, with portable ExecuTorch as the final fallback.
These are peer delegates applied in partitioner order; "stacking" refers to
that ordered partitioning, not one runtime wrapping another.

Terminology
-----------

``Edge exporter``
  The Python model decomposition and TensorRT compilation code under
  ``tools/hf/exporters``.

``Edge-LLM``
  The TensorRT-Edge-LLM project. It supplies plugins and C++ VLA runtime
  components such as ``VlaInferenceRuntime``, ``LLMEngineRunner``, and
  ``ActionRunner``.

``TensorRT delegate``
  The existing ExecuTorch ``TensorRTBackend``. Each lowered delegate owns one
  deserialized TensorRT engine and execution context.

``Edge-LLM delegate``
  The proposed out-of-tree ``EdgeLLMBackend``. One backend implementation
  creates component-specific handles backed by ``VitRunner``,
  ``LLMEngineRunner``, ``ActionRunner``, or later runner types.

``component``
  A separately compiled policy stage. PI0.5 currently has ``vision``,
  ``language``, and ``action`` components.

``orchestration backend``
  A possible later backend that owns an entire VLA request and several engines
  behind one delegate. This is distinct from the initial component-granular
  ``EdgeLLMBackend``.

Current state
-------------

Edge exporter
~~~~~~~~~~~~~

``EdgeExporter`` calls the selected ``EdgeSpec`` to prepare component bundles,
compiles every bundle, and exports a small runtime graph. PI0.5 produces the
following graph shape:

.. code-block:: text

   pixel_values
        |
        v
   edge_llm::execute_engine("vision")
        |
        v
   edge_llm::fuse_prefix
        |
        v
   edge_llm::execute_engine("language")
        |
        v
   edge_llm::execute_engine("action")
        |
        v
     velocity

The custom operator in ``tools/hf/exporters/ops.py`` accepts a filesystem path,
loads a ``TorchTensorRTModule``, and caches that Python module. This is useful
for exporter parity tests, but a path and Python module registry are not a
deployable ExecuTorch engine representation.

TensorRT ExecuTorch backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``torch_tensorrt.executorch.export`` recognizes
``tensorrt::execute_engine`` and
``tensorrt::no_op_placeholder_for_execute_engine``. It:

#. extracts serialized engines and metadata;
#. rewrites engine objects into export-safe buffers;
#. runs ``TensorRTPartitioner`` before any caller-provided partitioners;
#. serializes each TensorRT partition to a TR01 or TR02 blob; and
#. emits an ExecuTorch program containing one ``TensorRTBackend`` delegate per
   engine call.

TR02 additionally carries aliased-I/O metadata used for in-place KV updates.
The backend rejects engines requiring a TensorRT output allocator.

Multiple TensorRT engine calls are already supported. Each call pays a delegate
boundary cost, and a per-engine weight-streaming budget applies independently
to every engine.

TensorRT-Edge-LLM runtime
~~~~~~~~~~~~~~~~~~~~~~~~~

``VlaInferenceRuntime`` owns substantially more state than one TensorRT
delegate: component runners, tokenizer, embedding tables, shared TensorRT
context memory, KV caches, sampling workspaces, action state, and system-prompt
cache entries. Its constructors also consume an engine directory hierarchy.

The proposed backend lifts the component runners into ExecuTorch without
initially wrapping the whole ``VlaInferenceRuntime`` request API. For PI0.5,
the vision, language, and action partitions each create their corresponding
runner and pass graph values between delegates. This preserves component-level
partitioning while reusing Edge-LLM's runtime specialization.

Goals
-----

* Produce a libtorch-free PI0.5 ``.pte`` from the Edge exporter.
* Add an out-of-tree ``EdgeLLMPartitioner`` and ``EdgeLLMBackend`` for named
  Edge component operators.
* Encode the selected Edge runner, serialized engine, state requirements, and
  component metadata in each delegate payload.
* Reuse the existing TensorRT backend design and packed format where useful,
  while versioning Edge-specific metadata separately.
* Preserve the component boundaries and numerical behavior established by the
  Edge exporter and TensorRT-Edge-LLM reference runtime.
* Support composition with portable ExecuTorch and CUDA/AOTI operations.
* Define explicit contracts for engine metadata, mutable state, plugins,
  streams, packaging, and build integration.
* Keep the generic ``TensorRTBackend`` path available as the vanilla baseline.

Non-goals
---------

* Export every VLA family in the first milestone.
* Match the complete JSON request interface of ``VlaInferenceRuntime``.
* Add concurrent execution of component engines.
* Define a stable public C ABI for all TensorRT-Edge-LLM runners.
* Replace the existing TensorRT delegate; generic engine nodes continue to use
  it.
* Hide the entire VLA behind one monolithic delegate in the first milestone.

Proposed architecture
---------------------

.. code-block:: text

   HuggingFace / LeRobot policy
                 |
                 v
       EdgeSpec decomposition
                 |
                 v
    TensorRT component compilation
      vision  language  action
          \      |      /
           \     |     /
            v    v    v
      ExecuTorch export bridge
     path ops -> named Edge ops
                 |
                 v
      to_edge_transform_and_lower
        |             |                 |
        v             v                 v
   TensorRT       EdgeLLM           optional CUDA
   Partitioner    Partitioner       Partitioner
        |             |                 |
        |        vision/language/       |
        |        action delegates       |
        +-------------+-----------------+
                      v
             .pte program
                   |
                   v
      libtorch-free ExecuTorch runner

The PI0.5 performance program is expected to contain three
``EdgeLLMBackend`` delegate instances. Each instance has its own stateful
handle, but all are created by the same backend implementation:

.. code-block:: text

   EdgeLLMBackend handle: vision
     -> VitRunner + vision engine + preprocessing state

   EdgeLLMBackend handle: language
     -> LLMEngineRunner + language engine + KV/profile state

   EdgeLLMBackend handle: action
     -> ActionRunner + action engine + rollout state

The vanilla program instead contains three ``TensorRTBackend`` delegates and
keeps component-specific behavior in the graph or calling application. Both
artifacts are useful: the vanilla path isolates export and packaging failures,
while the Edge path validates the intended optimized runtime.

Export artifact contract
~~~~~~~~~~~~~~~~~~~~~~~~

The component compiler should expose a structured artifact rather than making
the exported graph rediscover information from ``config.json``. The logical
artifact contains:

* component name and model type;
* Edge runner kind and runtime ABI version;
* serialized TensorRT plan bytes;
* ordered input and output binding names;
* device and target-platform metadata;
* hardware-compatibility and weight-streaming metadata;
* aliased-I/O declarations, when present; and
* runner configuration, state requirements, sidecar references, and diagnostic
  shape/dtype information.

The existing directory format remains available for Python execution and
TensorRT-Edge-LLM interoperability. The ExecuTorch bridge consumes the same
artifact in memory and creates the packed payload expected by
``EdgeLLMBackend``. The outer program must not retain an absolute engine path in
its executable graph.

The bridge should be an explicit API, for example an Edge export mode or a
separate ``to_executorch`` operation. It replaces the prototype
``edge_llm::execute_engine(path, component, tensors)`` calls with named,
export-safe component operators whose engine artifacts are available to
``EdgeLLMPartitioner``. Keeping path resolution out of both partitioners
preserves deterministic packaging and keeps ``TensorRTBackend`` generic.

Operator and partition contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generic engine calls continue to use the existing ``tensorrt`` operator schema.
``TensorRTPartitioner`` continues to recognize only:

* ``tensorrt::execute_engine``; and
* ``tensorrt::no_op_placeholder_for_execute_engine``.

The performance path defines named ``edge_llm`` component operators. The exact
schemas are an implementation decision, but the logical forms are:

.. code-block:: text

   edge_llm::vision_tower(inputs, component_artifact) -> outputs
   edge_llm::language_runtime(inputs, mutable_state, component_artifact) -> outputs
   edge_llm::action_runtime(inputs, mutable_state, component_artifact) -> outputs

``EdgeLLMPartitioner`` claims only the supported named operators, assigns the
``EdgeLLMBackend`` delegation tag, and gives backend preprocessing one
component per partition. Backend preprocessing validates the operator kind
against the artifact, copies the serialized plan and runtime metadata into a
versioned packed payload, and rejects unknown runner kinds.

Non-component operations such as ``fuse_prefix`` and
``scatter_image_tokens`` still need an ExecuTorch lowering strategy:

#. prefer portable operators when their semantics and performance are
   sufficient;
#. otherwise let a caller-provided ``CudaPartitioner`` claim them; or
#. add a narrowly scoped native custom kernel when neither route is viable.

The Torch-TensorRT export pipeline currently runs ``TensorRTPartitioner`` first
and caller-provided partitioners afterward. The Edge partitioner must be placed
before the CUDA catch-all:

.. code-block:: text

   TensorRTPartitioner
   EdgeLLMPartitioner
   CudaPartitioner
   portable fallback

The first partitioner does not steal Edge operators because it recognizes only
the ``tensorrt`` schemas. A PI0.5 performance artifact should contain three
``EdgeLLMBackend`` delegate instances plus only the expected residual
portable/CUDA partitions. The vanilla artifact should contain three
``TensorRTBackend`` delegates.

State and KV-cache contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each lowered Edge partition creates an ``EdgeLLMHandle`` whose lifetime matches
the loaded ExecuTorch method. The handle contains:

* a validated component/runner kind;
* one Edge runner instance and its TensorRT engine/context;
* component configuration and shared execution-memory references;
* persistent runtime state such as KV cache, profile state, tokenizer data, or
  action rollout state when that component owns it; and
* synchronization needed to prevent overlapping use or destruction.

For PI0.5:

* the vision handle owns ``VitRunner`` and image-preprocessing buffers;
* the language handle owns ``LLMEngineRunner``, optimization-profile state, and
  the required KV-cache representation; and
* the action handle owns ``ActionRunner``, noise/timestep state, and rollout
  configuration.

Cross-component values such as vision embeddings and PI0.5 prefix K/V remain
explicit graph values unless the selected Edge API requires shared state.
Graph-visible mutable state must use ExecuTorch mutation semantics; state owned
entirely inside a runner must be reset and scoped through an explicit backend
API rather than hidden global variables.

The packed payload may reuse TR01/TR02 engine and binding fields, but Edge state
metadata needs its own versioned extension. KV aliasing must agree with both the
packed metadata and the engine/runner API. Device-resident state cannot be
silently staged through host scratch.

Native backend and runner adapter contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``EdgeLLMBackend`` implements the standard ExecuTorch
``BackendInterface`` lifecycle:

``init``
  Parse and version-check the packed payload, validate the component operator,
  ensure plugin registration, select the CUDA device, construct the selected
  runner, allocate persistent state, and return an ``EdgeLLMHandle``.

``execute``
  Convert ExecuTorch ``EValue`` arguments into the runner's tensor views, apply
  component-specific validation, pass the shared caller stream to the runner,
  execute the component, and expose outputs/state mutations back to
  ExecuTorch.

``destroy``
  Wait for in-flight work, destroy the runner and TensorRT objects, and release
  backend-owned resources without racing another ``execute``.

The logical handle shape is:

.. code-block:: cpp

   struct EdgeLLMHandle {
     ComponentKind kind;
     DeviceId device;
     ComponentMetadata metadata;
     std::variant<
         std::unique_ptr<VisionRunnerAdapter>,
         std::unique_ptr<LanguageRunnerAdapter>,
         std::unique_ptr<ActionRunnerAdapter>>
         runner;
     PersistentState state;
     Synchronization synchronization;
   };

This is one backend implementation with multiple handle instances, not one
backend class per model component.

Current Edge-LLM runner constructors consume engine directories and sidecar
files. The ExecuTorch backend must not reconstruct exporter-machine paths or
write temporary engine directories. Add narrow adapters or runner overloads
that accept:

* an in-memory engine byte view;
* parsed component configuration;
* named sidecar data;
* externally owned input/output tensor views; and
* the ExecuTorch caller's CUDA stream.

The adapters should reuse Edge-LLM algorithms rather than duplicate them in
Torch-TensorRT. They should also separate factory code from base runners. For
example, the PI0.5 vision adapter should construct ``VitRunner`` directly
instead of linking the all-model ``MultimodalRunner::create`` factory.

Engine and sidecar packaging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Component TensorRT plans, runner selection, binding metadata, and compact
configuration are embedded in ``.pte`` delegate blobs. The backend reconstructs
an object that owns both the Edge runtime state and the engine.

Large non-engine data has three possible homes:

* graph constants for small immutable tensors;
* ExecuTorch named external data (``.ptd``) for large tensor sidecars; or
* runner-provided named files for data owned by an Edge component.

The first component smoke test may accept prepared tensors to isolate backend
bring-up. The complete PI0.5 performance path requires a NamedDataMap or
versioned external-bundle contract for every tokenizer, embedding, config, or
other file consumed by the selected runners.

No production artifact may depend on an absolute exporter-machine path.
External data must be addressed by a logical name and validated with a version,
size, and integrity check.

Plugin contract
~~~~~~~~~~~~~~~

TensorRT-Edge-LLM plugin layers are inside the component engines; they are not
operators in the outer ExecuTorch graph. Their plugin creators must be
registered before any containing engine is deserialized.

The runner package must:

* ship a compatible ``libNvInfer_edgellm_plugin.so``;
* load and initialize it before loading the ``.pte`` module;
* report a clear error when the library or required creator/version is absent;
  and
* keep plugin and engine build revisions in the deployment manifest.

The Edge backend also links the runtime libraries that provide its selected
runners. Plugin registration and runner linkage are separate concerns: the
plugin library supplies TensorRT layer creators, while the runner library
supplies C++ orchestration and state.

CUDA stream and execution contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All CUDA-capable delegates in one process must observe the same caller-stream
thread-local state. The runner uses the backend-neutral guard:

.. code-block:: cpp

   #include <executorch/extension/cuda/caller_stream.h>

   executorch::extension::cuda::CallerStreamGuard guard(stream);
   module.forward(inputs);

The selected stream must belong to the component engines' device. Every
``EdgeLLMBackend`` operation passes this stream to the Edge runner API. Calls
using one delegate handle must not overlap each other or that handle's
destruction. The Edge handle needs the same mutex and in-flight completion
discipline as ``TensorRTBackend`` unless the underlying runner provides a
stronger contract.

When all bindings are device, managed, or unified memory, execution may return
while TensorRT work is still in flight. Callers must:

* keep directly bound inputs and outputs alive and unmodified;
* order cross-stream producers and consumers with their own events; and
* synchronize before reading an output on the host.

Host staging and runner operations that consume host results take the
synchronized path. With no active guard, the backend uses
``cudaStreamPerThread``.

This contract permits one green-context stream to drive both TensorRT and
CUDA/AOTI delegates. Green-context execution is currently a manually validated
configuration, not a complete CI guarantee.

Runner and build integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The packaged backend lives in ``libtorchtrt.tar.gz`` under
``torch_tensorrt/src/torch_tensorrt/executorch``. It requires ExecuTorch 1.4 or
a source revision containing the required caller-stream and CUDA extension
changes documented in that package's README.

Three runner configurations are relevant:

``TensorRT plus CUDA/AOTI``
  Add ExecuTorch with ``EXECUTORCH_BUILD_CUDA=ON`` and link
  ``torchtrt::executorch_backend``. Both delegates use ExecuTorch's shared
  ``extension_cuda``.

``TensorRT-only, libtorch-free``
  Leave the full CUDA/AOTI backend disabled. The Torch-TensorRT package builds
  the minimal shared caller-stream ``extension_cuda`` library from the same
  ExecuTorch checkout.

``Edge-LLM performance runtime``
  Link ``EdgeLLMBackend`` and the Edge runtime libraries that provide the
  selected component runners. Load the Edge plugin library before any component
  engine is deserialized. Optionally link the CUDA/AOTI backend for residual
  graph partitions.

On ELF platforms a prebuilt
``EXECUTORCH_EXTENSION_CUDA_LIBRARY`` may be supplied, but it must be a shared
object. Static copies are invalid because each delegate would receive distinct
caller-stream TLS state.

The TensorRT backend archive is exposed as both the
``torchtrt::executorch_backend`` consumer target and
``executorch_trt_backend`` build target. There is no separate backend build
step when the runner links the consumer target.

TensorRT-Edge-LLM itself is CMake-based and currently exposes a broad
``edgellmCore`` static target plus ``libNvInfer_edgellm_plugin.so``. The
Torch-TensorRT Bazel build should invoke that CMake project through
``rules_foreign_cc`` and produce an out-of-tree Edge runtime library consumed
by the ExecuTorch runtime package.

The first build may link ``edgellmCore`` coarsely, as proposed in the meeting,
to establish correctness. The follow-up build split should expose narrow
targets such as:

.. code-block:: text

   edgellm_multimodal_base
   edgellm_vit_runner
   edgellm_language_runner
   edgellm_action_runner
   edgellm_tokenizer
   NvInfer_edgellm_plugin

PI0.5 then links only the runners and support libraries represented by its
operators. The all-model multimodal factory must be separated from the base
runner because it references every concrete model runner. Shared libraries are
deployment units and are not made smaller merely because only one API is
called; use split targets or static archives with section garbage collection
and LTO when binary size matters.

Phased implementation
---------------------

Phase 0: lock both reference paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Establish reproducible PI0.5 reference outputs before changing packaging.

Deliverables:

* a pinned model, exporter, Torch-TensorRT, TensorRT, and plugin revision;
* eager, Edge-exported TensorRT, and TensorRT-Edge-LLM reference outputs;
* a vanilla ExecuTorch artifact using generic ``TensorRTBackend`` delegates;
* per-component parity metrics and representative Libero inputs; and
* a recorded engine I/O and optimization-profile contract.

Exit criteria:

* vision, language, and action component parity meet agreed tolerances;
* the same inputs are usable by Python and C++ references; and
* failures can be localized to export, generic delegate packaging, or one Edge
  component.

Phase 1: Edge vision delegate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement the out-of-tree backend skeleton and lower one
``edge_llm::vision_tower`` operator to a ``VitRunner``-backed delegate. Use an
image-only or prepared-tensor mode first if necessary to keep tokenizer
packaging outside this bring-up milestone.

Deliverables:

* named vision operator with fake/meta behavior;
* ``EdgeLLMPartitioner`` and versioned backend payload;
* ``EdgeLLMBackend`` registration, ``init``, ``execute``, and ``destroy``;
* one stateful handle that constructs a narrow ``VitRunner`` adapter;
* Bazel-to-CMake Edge-LLM build and plugin packaging;
* one-Edge-delegate ``.pte``;
* libtorch-free C++ runner invocation; and
* Python and C++ integration tests.

Exit criteria:

* the serialized program contains one ``EdgeLLMBackend`` delegate;
* backend initialization selects only the vision runner kind;
* it loads without Python or libtorch;
* plugin registration occurs before engine deserialization;
* output matches the Edge-LLM/Edge-exporter vision reference;
* execution uses the ExecuTorch caller stream; and
* no absolute build-machine paths appear in the artifact.

Phase 2: PI0.5 component pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add the PI0.5 language and action operators. The program passes vision
embeddings, compact prefix values, and language outputs between three
component-granular Edge delegates.

Deliverables:

* vision, language, and action Edge operator schemas;
* three packed component payloads and stateful handles;
* ``VitRunner``, ``LLMEngineRunner``, and ``ActionRunner`` adapters;
* portable or CUDA lowering for prefix fusion;
* one caller stream across all delegates;
* fixed-shape or bounded-dynamic Libero sample coverage; and
* per-stage and end-to-end parity diagnostics.

Exit criteria:

* the program contains three ``EdgeLLMBackend`` delegates and only the
  expected residual partitions;
* each payload selects the intended runner and rejects mismatched metadata;
* execution is libtorch-free;
* cross-component binding order and ownership are validated;
* end-to-end output matches the Edge-exported reference; and
* repeated runs do not leak, corrupt, or unintentionally retain state.

Phase 3: complete stateful PI0.5 runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Match the required PI0.5 behavior from the Edge-LLM C++ runtime rather than
stopping at one invocation of each engine.

Deliverables:

* language prefill/decode profile control through ``LLMEngineRunner``;
* explicit KV-cache ownership, reset, and persistence semantics;
* action noise, timestep schedule, and rollout ownership through
  ``ActionRunner``;
* tokenizer, embedding, and runner-config packaging;
* shared-context-memory and synchronization contract; and
* memory and weight-streaming budgets across all components.

Exit criteria:

* profile transitions are tested and observable;
* KV and action state persist or reset exactly when requested;
* device-resident asynchronous execution has an end-to-end test;
* host and device I/O paths produce equivalent results; and
* sidecar lookup is location-independent and version-checked.

Phase 4: dependency refinement and family expansion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Replace the coarse ``edgellmCore`` dependency with narrow component targets and
measure the result against ``VlaInferenceRuntime``. Then reuse the contracts
for GR00T or the next selected family.

Deliverables:

* split runner, tokenizer, kernel, and factory targets;
* a dependency manifest for each supported operator set;
* binary-size, load-time, memory, and latency comparisons;
* model-family versioning for payload metadata; and
* a decision on whether any measured boundary requires a whole-request
  orchestration backend.

Exit criteria:

* a PI0.5 build does not link unrelated Qwen, InternVL, Phi, Gemma, Nemotron,
  audio, or builder runtimes;
* required plugins and kernels remain registered;
* the optimized Edge path improves the metric that justified it; and
* adding another family does not change the PI0.5 payload contract.

Validation strategy
-------------------

Unit tests
~~~~~~~~~~

* Artifact construction rejects missing plans, unknown runner kinds, duplicate
  binding names, incompatible state declarations, and unsupported versions.
* The bridge preserves component order, runner kind, binding metadata, device,
  state, and sidecar references.
* The outer exported graph contains no path-based engine call after bridging.
* Edge partitioning produces exactly one partition per named component
  operator, while TensorRT partitioning ignores those nodes.
* Backend preprocessing rejects a vision operator paired with language/action
  metadata and equivalent mismatches.
* Residual graph operations are claimed by the expected portable/CUDA path.
* The versioned Edge payload round-trips engine, runner, state, and sidecar
  metadata.

Integration tests
~~~~~~~~~~~~~~~~~

* Edge vision export to ``.pte`` to C++ ``VitRunner`` execution.
* Full PI0.5 graph with three ``EdgeLLMBackend`` delegate IDs and numerical
  parity.
* Parallel vanilla artifact with three ``TensorRTBackend`` delegates as a
  differential baseline.
* Missing or incompatible Edge-LLM plugin library produces a diagnostic error.
* One process resolves exactly one shared ``libextension_cuda.so``.
* Edge, TensorRT, and CUDA/AOTI delegates observe the same caller stream.
* KV state persists and resets according to the language runtime contract.
* Action rollout produces the same seeded trajectory as the Edge reference.
* Handle teardown waits for backend-owned in-flight work.

Performance and reliability tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Record component latency, delegate-boundary overhead, peak device memory,
engine load time, runtime-library size, and artifact size. Compare the vanilla
TensorRT-delegate program, the Edge-delegate program, the Edge exporter Python
path, and ``VlaInferenceRuntime``.

The existing host-staged reference-runner smoke test does not validate
device-resident asynchronous return. Add a dedicated test that keeps inputs and
outputs on device and verifies lifetime and cross-stream ordering. Also track
green-context execution with the internal completion event as a separate
coverage gap until automated.

Failure handling
----------------

Export must fail before writing a deployable artifact when:

* an Edge call cannot be converted from a path to a named component artifact;
* an operator, runner kind, and payload component disagree;
* binding names or counts do not match graph inputs and outputs;
* mutable-buffer semantics cannot be represented;
* a requested residual op has no portable or delegated implementation; or
* sidecar data has no location-independent package representation.

Runtime initialization must identify the component and delegate when plugin
registration, runner construction, engine deserialization, device selection,
metadata parsing, state allocation, or binding validation fails.

Risks and mitigations
---------------------

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Risk
     - Impact
     - Mitigation
   * - Path-based ``edge_llm`` operators
     - ``.pte`` is not self-contained
     - Bridge to named Edge operators with embedded component payloads
   * - Wrong runner selected
     - Engine bindings or state are interpreted incorrectly
     - Version component kind and validate it against the operator and metadata
   * - Language runtime state is hidden
     - Prefill/decode or KV reuse becomes nondeterministic
     - Give each language handle explicit profile, reset, and KV ownership APIs
   * - Missing plugin creator
     - Engine deserialization fails
     - Version and initialize the plugin library before module load
   * - Incorrect KV ownership
     - State silently freezes or updates staging memory
     - Validate packed aliases and distinguish graph-owned from runner-owned KV
   * - Delegate boundary overhead
     - Component-granular Edge path misses latency target
     - Measure against ``VlaInferenceRuntime`` before considering one delegate
   * - Multiple caller-stream TLS copies
     - Delegates execute on different streams
     - Require one shared ``libextension_cuda`` and test dynamic linkage
   * - Sidecar files depend on exporter paths
     - Artifact cannot be relocated
     - Use named external data with integrity metadata
   * - Edge-LLM runtime paths diverge
     - PI0.5, GR00T, and Alpamayo acquire incompatible assumptions
     - Stabilize PI0.5 contracts first and version family-specific metadata
   * - Build-system mismatch
     - Runner depends on private Edge-LLM source layout
     - Use a pinned CMake external build, then expose installable narrow targets
   * - Broad ``edgellmCore`` linkage
     - PI0.5 ships unrelated model runners and kernels
     - Permit it for bring-up, then split factories and component libraries

Rejected initial alternatives
-----------------------------

Teach ``TensorRTPartitioner`` to load engine paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rejected because partitioning should not perform filesystem-dependent engine
resolution. It would make ``.pte`` creation depend on mutable external state
and would duplicate the existing engine extraction and serialization path.

Wrap all of ``VlaInferenceRuntime`` in one delegate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deferred because it hides component boundaries and combines packaging,
tokenization, orchestration, and all engines before the component backend
contract is tested. It remains a later optimization if measured delegate
boundaries require it.

Use only generic TensorRT delegates for the performance path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retained as the vanilla baseline, but rejected as the Edge performance design
because it does not lift ``VitRunner``, ``LLMEngineRunner``, ``ActionRunner``,
or their persistent runtime semantics into ExecuTorch.

Compile the complete policy into one TensorRT engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rejected as the default because the component runtimes have different profile,
state, and plugin requirements, and graph-level orchestration is intentionally
visible in the Edge exporter.

Open decisions
--------------

The following decisions must be closed before their associated phase begins:

* Phase 1: exact named operator schemas and initial packed-payload version.
* Phase 1: direct runner APIs versus narrow ExecuTorch adapter classes.
* Phase 1: whether the first Edge external build returns a static archive, a
  shared runtime library, or both.
* Phase 2: portable versus CUDA implementation of ``fuse_prefix``.
* Phase 2: ownership of shared TensorRT context memory across component handles.
* Phase 3: whether decode/action rollout occurs inside one backend call or
  across repeated ExecuTorch method calls.
* Phase 3: NamedDataMap/``.ptd`` versus a versioned external bundle for
  tokenizer and embedding data.
* Phase 4: criteria for replacing component delegates with one orchestration
  delegate.

Workstreams
-----------

``Exporter and packaging``
  Structured engine artifacts, op bridge, sidecar manifest, and user API.

``ExecuTorch lowering``
  Named operators, ``EdgeLLMPartitioner``, payload preprocessing, partitioner
  composition, and mutation declarations.

``Native runtime``
  ``EdgeLLMBackend``, component handle lifecycle, runner dispatch,
  caller-stream behavior, state ownership, and diagnostics.

``Edge-LLM integration``
  Bazel/CMake external build, plugin/runtime revision pinning, runner adapters,
  narrow library targets, and reference behavior.

``End-to-end validation``
  PI0.5 fixtures, parity thresholds, C++ runner tests, performance measurement,
  and Libero demonstration.

Source map
----------

The main implementation and reference points are:

* ``tools/hf/exporters/exporter.py`` -- Edge export orchestration.
* ``tools/hf/exporters/compile.py`` -- component compilation and disk artifact.
* ``tools/hf/exporters/ops.py`` -- current path-based outer custom operators.
* ``tools/hf/exporters/models/pi05/spec.py`` -- PI0.5 decomposition and wiring.
* ``py/torch_tensorrt/executorch/_export.py`` -- ExecuTorch export pipeline.
* ``py/torch_tensorrt/executorch/partitioner.py`` -- TensorRT partitioning.
* ``py/torch_tensorrt/executorch/backend.py`` -- engine blob preprocessing.
* ``cpp/src/torch_tensorrt/executorch/TensorRTBackend.cpp`` -- native backend.
* ``cpp/src/torch_tensorrt/executorch/README.md`` -- runner and stream contract.
* TensorRT-Edge-LLM ``cpp/multimodal/vitRunner.*`` -- PI0.5 vision runner.
* TensorRT-Edge-LLM ``cpp/runtime/vlaInferenceRuntime.*`` -- VLA reference.
* TensorRT-Edge-LLM ``cpp/runtime/llmEngineRunner.*`` -- language and KV
  reference.
* TensorRT-Edge-LLM ``cpp/action/actionRunner.*`` -- action runtime reference.
* TensorRT-Edge-LLM ``cpp/CMakeLists.txt`` -- current broad runtime/plugin
  targets to integrate and later decompose.
