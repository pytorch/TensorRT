# Implementation plan: user-selected PyTorch regions

Status: an initial implementation is now present. See the
[MVP usage and limits](py/torch_tensorrt/region/README.md). The sections below
describe the implementation contract and its remaining qualification work.

This plan defines the first MVP: a user marks operations with
`torch_tensorrt.region.execute_in_torch()`, and Torch-TensorRT keeps those
operations together in PyTorch while compiling eligible surrounding operations.
It supersedes the callable-first frontend direction in
[the earlier design](region_annotations_design.md) for this milestone.

The first deliverable is a correctly captured and placed region. Pattern
matching, QDP replacement, kernel generation, tuning data, and agents are later
work and are not dependencies of this MVP.

## 1. User-visible result

~~~python
class MyModule(nn.Module):
    def __init__(self, attn, mlp):
        super().__init__()
        self.attn = attn
        self.mlp = mlp

    def forward(self, x):
        with torch_tensorrt.region.execute_in_torch(name="attention"):
            x = self.attn(x)
        return self.mlp(x)


compiled = torch_tensorrt.compile(
    model.eval().to(example_cuda_tensor.device),
    inputs=[example_cuda_tensor],
    ir="dynamo",
    strict=True,
    offload_module_to_cpu=False,
    min_block_size=1,
)
~~~

`name` is optional; the user's original `execute_in_torch()` form must work.
The example uses the planned initial capture configuration, subject to the
capture qualification in milestone 0. `min_block_size=1` makes placement tests
predictable; it is not a new requirement or default for ordinary compilation.

Expected behavior:

~~~text
input -> captured attention child in PyTorch -> eligible MLP in TensorRT -> output
~~~

The context has normal eager semantics. During compilation, its captured tensor
operations form one region. PyTorch execution remains on the declared device,
normally CUDA. It does not restore Python hooks or side effects lost in export.
The body still has to be exportable; this API is not an escape from tracing.

Only the annotated invocation is forced into PyTorch. Another call to the same
module outside the scope remains eligible for TensorRT. Surrounding operations
still obey normal converter support and minimum partition size. A region can
share a larger PyTorch partition with neighboring unsupported operations.

## 2. MVP scope and explicit limits

| Area | Initial contract |
| --- | --- |
| Entry point | Torch-TensorRT-owned `torch_tensorrt.compile(..., ir="dynamo")` capture |
| Public API | `execute_in_torch(*, name: str | None = None)` |
| Workload | Eval-mode, functional inference; first GPU qualification uses static shapes on one CUDA device |
| Boundary | Tensor inputs, read-only Tensor state, Tensor or flat tuple-of-Tensor outputs; literals may remain in the child |
| Scopes | Anonymous, named, adjacent, repeated, and Python-unrolled occurrences |
| Partitioners | Fast and global, including automatic fast-to-global fallback |
| Empty/dead pure scope | Remove markers and report a no-op |
| Runtime | Captured FX/ATen child, with no region marker or capture session required |

Reject nested/overlapping scopes, graph breaks inside a scope, mutation,
escaping input/output aliases, RNG effects, unsupported control-flow bodies,
and runtime scalar boundaries. Python variable reassignment such as
`x = self.attn(x)` is allowed; in-place Tensor mutation is a different operation.
Disconnected pure computations inside one scope are allowed when their full
input/output boundary is representable.

Defer dynamic shapes, training, mutable state/KV caches, resource/hierarchical
partitioning, refit/weight streaming, cross-compilation, and the
`torch.compile` backend. Do not add contiguous-layout or single-output
restrictions merely because a future kernel provider might require them.

For graphs containing nonempty regions:

- Reject `require_full_compilation=True` and engine-only output.
- Require `offload_module_to_cpu=False` so PyTorch state stays on its device.
- Reject `enable_autocast=True` before calibration until child-aware support exists.
- Reject resource partitioning and other deferred combinations explicitly.
- Preserve existing behavior for models without regions.

Annotations are active only during the owned capture session. In eager execution
and unrelated compiler sessions they are no-ops. Exporting the model separately
can therefore erase the annotation; compiling such an ExportedProgram is not a
supported way to use this MVP. Document the direct entry point clearly.

## 3. End-to-end implementation

~~~text
execute_in_torch scope during owned capture
    -> paired scope markers + exact membership tags
    -> validate and extract a child graph at the first safe capture seam
    -> preserve the child behind a region higher-order operator (HOP)
    -> normalize/decompose parent and child
    -> materialize one opaque PyTorch child call
    -> preserve that call through parent graph rewrites
    -> existing support analysis and fast/global partitioning
    -> audit placement and build surrounding TensorRT engines
~~~

The markers are temporary capture bookkeeping. The HOP is a temporary container
that carries the child through decomposition. The final compiled parent calls
an ordinary PyTorch child module. Neither temporary mechanism remains at runtime.

## 4. Milestone 0: prove the context boundary survives capture

This is the first engineering task and a prerequisite for enabling the API.
A normal context manager does not establish an FX boundary. Previous callable
HOP experiments do not prove that this scope-based mechanism works.

Build a small capture harness around this proposed protocol:

~~~text
begin(optional_name) -> fresh capture sentinel
    body operations receive one graph-local membership token
end(the_same_sentinel)
~~~

The sentinel pairs an end with its exact begin. The token identifies the body
operations. Ordered markers alone do not prevent pure operations moving across
them, so both kinds of evidence are required. A name is a diagnostic label,
never a membership identifier.

Tasks:

1. Probe internal ordered-effect marker operators and primitive tracing metadata
   on each PyTorch version/export mode proposed for support. Test strict and
   non-strict separately.
2. Prove a token mechanism survives adjacent scopes, retracing, tracing rollback,
   exceptions, and object reuse. Sentinel-derived identity is a candidate to
   test, not an assumed solution. Do not use a Python-global occurrence counter.
3. Prove the compile-session check itself is trace-compatible. A thread-local
   or context-local accessor inside `forward` is not automatically traceable.
4. Give the no-Tensor-argument `begin` operator a working catch-all implementation
   and compatible fake behavior. A CPU-only dispatch registration is insufficient.
   Sentinels must not read tensor data, select a CUDA device, or modify model state.
5. Identify the first graph Torch-TensorRT can safely normalize. Verify that
   markers and tags still exist there and that effect-token handling has not
   changed the exported model's input/output contract.
6. Outline a small region into the experimental `hints_wrapper` carrier and
   verify that decomposition preserves its child, explicit state, and outputs.

The intended normalization seam is the very start of `pre_export_lowering()`.
However, export has its own internal transformations. If that point is too late,
the spike must establish an earlier capture hook or a tested direct scope-to-HOP
adapter. Do not repair an unknown effect-token signature by simply deleting nodes.

Deliverables: reproducible tests, raw/normalized graph examples, a supported
version/mode table, and the selected marker/token/carrier implementation.

Exit gate: exact membership, valid exported signatures, and a durable child
boundary are demonstrated. If any scope can silently disappear or become
ambiguous, keep public support disabled and resolve the capture mechanism first.

## 5. Milestone 1: API and owned compilation session

Add the public context manager and import it through `torch_tensorrt.region`.
Expose it with the Dynamo frontend, without requiring QDP/kernel-generation
features.

The public compile entry creates a session before tracing and keeps it through
compilation. It records capture mode and compatibility settings and owns the
region records. It always cleans up after success or failure.

During owned capture, `__enter__` starts the proven scope protocol and
`__exit__` closes it. Do not suppress user exceptions. A graph break inside a
region becomes an actionable capture error rather than partial eager execution.

Use explicit `strict=True` for the initial qualified path if only strict export
passes milestone 0. Keep the existing non-strict default for unannotated models.
When a scope is entered under an unsupported mode, raise immediately; do not
silently retry under another mode or wait until tags might have disappeared.
Broaden support only when the same capture tests pass in that mode.

Assign occurrence IDs from the accepted graph, such as `attention#0` or
`region#1`. Repeated labels are legal. No occurrence state is incremented by
Python tracing. Concurrent and repeated compilations must not share region plans.

Exit gate: eager behavior and exceptions are unchanged; the owned capture path
emits regions; unrelated backends receive no marker operators; session state is
restored after failures.

## 6. Milestone 2: extraction and preservation through lowering

At the safe seam established by milestone 0, `normalize_region_scopes()`:

1. Pairs begin/end using the sentinel edge.
2. Checks that membership tags and marker intervals agree. Reject orphaned,
   crossed, nested, duplicate-token, or conflicting scopes; never guess.
3. Selects body operations and computes all external inputs and externally used
   outputs. Treat parameter/buffer reads as explicit state inputs.
4. Checks purity, aliasing, supported boundary types, and whether extraction
   would create a dependency cycle.
5. Outlines a child GraphModule using FX extraction utilities, preserving tensor
   metadata and original-to-child input/output mappings.
6. Replaces the body with a region HOP, reconnects outputs, and removes every
   marker and sentinel use. Validate graph and exported signatures.

Example:

~~~text
Before:
    h = pre(x)
    begin
    a = h @ weight
    y = relu(a + bias)
    end
    out = post(y) + h

After extraction:
    h = pre(x)
    y = region_hop(region_body, (h, weight, bias))
    out = post(y) + h

region_body(h, weight, bias):
    return relu(h @ weight + bias)
~~~

The residual `h` remains a parent dependency. Additional escaping results become
additional child outputs; internal values with no external users stay internal.
Use the extractor's deterministic input mapping for replay. This MVP has no QDP
schema and does not need the callable API's public-argument ordering machinery.

Store a compact compile-local record:

~~~text
RegionRecord:
    occurrence ID and optional name
    source location and route = force_torch
    private normalized reference child
    ordered input/state bindings and outputs
    expected parent leaf and final partition owner
~~~

Run required pre-lowering cleanup recursively on children and then the parent.
The current pass manager does not do this automatically. Normalize markers
before autocast calibration, not merely at index zero of the registered passes.
Decompose with the region boundary intact; rediscover HOPs afterward because
decomposition can return new graph objects.

After `ep.module()` and existing state handling, replace each HOP with a uniquely
named `call_module` and preserve output reconstruction. Keep state on its
declared device. The child is already normalized; parent TensorRT rewrites must
not descend into it or move operations across its boundary.

Protect the call from constant folding with a mandatory region check, not a
normal exclusion users can disable. Validate occurrence count, target, route,
and boundary after each parent rewrite. An unchanged region ID attached to a
different operation, such as a folded constant, does not satisfy the invariant.

Exit gate: original and extracted computations agree; child state and residual
connections remain correct; no marker survives; decomposition and post-lowering
preserve every nonempty region.

## 7. Milestone 3: enforce PyTorch placement

Teach both support testers to recognize a verified force-Torch child before
ordinary converter lookup, record the reason `explicit execute_in_torch region`,
and return unsupported for that call.

Also update the surrounding compiler logic:

- Count marked `call_module` regions as unsupported computational calls in
  `get_graph_converter_support_overview()`. It currently counts only
  `call_function` nodes.
- Reject conflicting full-compilation settings before early returns and direct
  whole-graph conversion. Ensure `skip_fusion` and `assume_full_support` cannot
  bypass the force-Torch policy.
- Keep minimum-size rules unchanged. Small neighboring TensorRT candidates may
  remain in PyTorch; this MVP introduces no minimum-size exemption.
- Skip force-Torch GraphModules during engine conversion in both fast and
  global paths. The existing conversion-loop skip is primarily fast-path logic.
- Preserve live parent state during cleanup as well as offload. The compiler
  currently removes parent `_frozen_param*` attributes before conversion; retain
  any attribute still referenced by a PyTorch region or other parent consumer.
  Include constants folded outside a scope and passed into it as inputs.
- After partitioning, find every occurrence recursively and verify that its
  child exists exactly once and is absent from every accelerated partition.
- Run the same placement/report checks on all early returns and after a
  fast-partitioner failure falls back to the global partitioner.

Carry an explicit placement manifest through these stages. Node and child
metadata identify candidates, but must be checked against the manifest rather
than treated as independent authority.

Use a test with TensorRT-supported operations inside the scope. Otherwise a
test might pass merely because those operations already lacked converters.

Exit gate: numerical parity and correct ownership under both partitioners for
`TensorRT -> PyTorch region -> TensorRT`, including branches and reused modules.
Verify actual engines exist for eligible neighbors, not just partition names.

## 8. Milestone 4: diagnostics and one supported save/load path

Report name/occurrence, source, input/output summary, explicit PyTorch policy,
and final owner. Extend existing fallback reporting to include region child
calls and distinguish user policy from missing converter support.

Dry-run captures and analyzes regions without building engines. Policy conflicts
appear in its report; malformed boundaries remain errors. A normal compile with
the same conflicts fails before placement shortcuts. Dry-run stops or skips
conflicting calibration/offload/resource paths after reporting them; it must
not execute an unsupported combination to predict its result. No sample
dataloader or provider invocation is needed anywhere in this MVP.

Saving needs separate qualification. The current exporter can inline ordinary
`_run_on_gpu` modules, but region children may be nested under them or remain
direct children with different names. Plain re-export may also erase hierarchy.

Initially qualify the static save path with `output_format="exported_program"`,
`retrace=False`, and `use_legacy_exporter=True`. Pin the last option explicitly:
`retrace=False` alone still permits an override to a re-export path.

1. Work on a serialization clone of the compiled module.
2. Extend export handling to recursively materialize/inline region children,
   including correct state references, without inserting their operations into
   existing TensorRT engine calls.
3. Preserve a minimal serializable provenance record and verify no temporary
   marker/HOP or capture-session object is required.
4. Save, load in a fresh process, and compare numerical results and devices
   under both partitioners. Inspect that engine boundaries remain intact.

If the selected path needs additional work, gate it explicitly until these
tests pass. Reject other unqualified save modes for region-bearing modules with
an actionable message. Do not silently drop the PyTorch computation.

The reload guarantee is correct hybrid execution. The exact child hierarchy may
be flattened in the saved graph. Recompiling a loaded artifact while preserving
the original annotation policy requires a durable region manifest and is deferred.

Exit gate: the documented save/load path works independently of the source
model's context manager and compile session. Unsupported modes fail clearly.

## 9. Code changes

All region paths below are new; existing paths identify integration seams.

| Location | Planned responsibility |
| --- | --- |
| `py/torch_tensorrt/region/__init__.py`, `_context.py` | Public context manager and eager behavior |
| `py/torch_tensorrt/region/_session.py`, `_marker_ops.py` | Owned capture state, proven marker protocol and compatibility checks |
| `py/torch_tensorrt/dynamo/regions/_capture.py` | Parse/validate scopes, compute boundary, outline, preserve HOP |
| `py/torch_tensorrt/dynamo/regions/_types.py`, `_resolve.py`, `_audit.py` | Region records, PyTorch child materialization, rewrite/placement invariants |
| [Public entry](py/torch_tensorrt/_compile.py), [package exports](py/torch_tensorrt/__init__.py), [tracer](py/torch_tensorrt/dynamo/_tracer.py) | Session lifetime, mode validation, API exposure; preserve/forward positional and keyword contracts |
| [Lowering entry](py/torch_tensorrt/dynamo/lowering/passes/_aten_lowering_pass.py), [pass manager](py/torch_tensorrt/dynamo/lowering/passes/pass_manager.py) | Earliest normalization, child recursion, parent rewrite audits |
| [Constant folding](py/torch_tensorrt/dynamo/lowering/passes/constant_folding.py) | Mandatory force-Torch folding barrier |
| [Compiler](py/torch_tensorrt/dynamo/_compiler.py), [support counts](py/torch_tensorrt/dynamo/partitioning/common.py) | Region resolution, policy checks, counts, early returns, conversion and audits |
| [Fast partitioner](py/torch_tensorrt/dynamo/partitioning/_adjacency_partitioner.py), [global partitioner](py/torch_tensorrt/dynamo/partitioning/_global_partitioner.py) | Node-local force-Torch classification and reason reporting |
| [Dry-run tracker](py/torch_tensorrt/dynamo/_DryRunTracker.py), [exporter](py/torch_tensorrt/dynamo/_exporter.py) | Region diagnostics and qualified persistence |
| `tests/py/dynamo/regions/` | Capture, extraction, partitioning, error and serialization coverage |

Do not introduce kernel-provider registries, pattern registries, or a new
partitioner. Future optimizers can consume the retained reference child and
boundary record.

## 10. Verification and release checklist

| Test group | Required evidence |
| --- | --- |
| Eager/session behavior | Same outputs/exceptions; no markers outside owned capture; cleanup and compile isolation |
| Membership | Named/unnamed, adjacent/repeated, unrolled calls, independent pure operations before/after the scope, tag/marker mismatches, exceptions and retracing |
| Extraction | Read-only state, residual fan-out, multiple inputs/outputs, independent computations, empty/dead scopes |
| Illegal bodies | Clear errors for nesting, graph breaks, mutation, alias escape, RNG and unsupported boundary/control flow |
| Lowering | Marker removal, preserved export signatures, recursive child cleanup, decomposition parity, constant-input region survives |
| Placement | Fast/global/fallback paths; only annotated invocation forced; actual neighboring engines; region body absent from engines; live parent constants survive cleanup |
| Compiler shortcuts | All-Torch graph, low supported-op count, minimum-size early return, full-support shortcuts, dry-run |
| Settings | Full-compilation/offload/autocast/resource/refit conflicts identified before unsafe work |
| Persistence | Qualified save/load path in fresh process; nested and direct children; state/devices/results preserved |
| Regression | Existing unannotated capture, partitioning, full-compilation, fallback-reporting and export tests unchanged |

Run capture/extraction tests without an engine build where possible. Run GPU
integration tests with `use_fast_partitioner=True` and `False`, initially FP32
and `min_block_size=1`; add the normal minimum-size setting to verify surrounding
placement remains ordinary. Compare the same eval-mode model/weights and inputs
before and after compilation with declared tolerances.

Suggested reviewable increments: capture spike; public session/API; extraction
and lowering; fast/global placement; diagnostics and persistence. Partitioning
work can use synthetic region children while capture is being qualified, but
end-to-end support cannot ship before the capture exit gate passes.

The MVP is complete when the user's context-manager example captures exactly
the marked operations, retains them in PyTorch, compiles eligible neighbors,
and passes the documented save/load and failure-mode tests. Kernel generation
can then build on a proven region boundary.
