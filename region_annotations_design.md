# Callable Regions in Torch-TensorRT

MVP direction update: the first implementation now targets the user-facing
`execute_in_torch()` context manager. See the
[MVP implementation plan](region_execute_in_torch_mvp_plan.md), which governs
that milestone. The callable API below remains a broader alternative design.

Status: design proposal. The public APIs and integration below are proposed;
existing FX capture, partitioning, engine building, and QDP registration are
reused.

A region is one invocation of a PyTorch callable, captured as a child FX graph.
The user chooses whether that invocation stays in PyTorch, uses an existing
TensorRT Quick Deployable Plugin (QDP), or requests a generated kernel.

The design has three rules:

1. The callable defines the boundary.
2. Torch-TensorRT resolves each region before ordinary partitioning.
3. Every successful kernel replacement becomes one QDP-backed operator.

This replaces the earlier context-manager proposal. There is no
`region.begin/end` protocol and no reconstruction of a `with` block from a
flat graph. Code previously inside a context manager moves into a function or
module passed to `invoke`. All routing decisions happen during compilation.

## 1. Public API

`invoke` is the canonical primitive. Its first two arguments are positional-only,
so region policy cannot consume keyword arguments intended for the body.

~~~python
def invoke(spec, reference_body, /, *args, **kwargs): ...

# Immutable policy specifications:
region.in_torch(name="attention")
region.qdp(name="activation", op=torch.ops.plugins.bias_gelu.default,
           on_failure="error")
region.generated(name="projection", provider="graphir", profile="default",
                 on_failure="partition")
~~~

| Policy | Successful compilation | Default failure policy |
| --- | --- | --- |
| `in_torch` | One opaque PyTorch child call | Capture/contract errors stop compilation |
| `qdp` | One call to the specified registered QDP overload | `error` |
| `generated` | One call to a generated, registered QDP operator | `partition` |

In ordinary eager execution, every policy simply calls
`reference_body(*args, **kwargs)`. In Torch-TensorRT-owned capture, the same
call becomes a region higher-order operator (HOP): an operator containing a
child graph. Unannotated code follows the existing compiler workflow.

For example, only this attention invocation stays in PyTorch:

~~~python
from torch_tensorrt import region

class Model(nn.Module):
    def forward(self, x, mask):
        h = self.pre(x)
        ctx = region.invoke(
            region.in_torch(name="attention"),
            self.attention, h, mask=mask,
        )
        return self.post(ctx)
~~~

`in_torch` keeps execution on the inputs' device, normally CUDA. It executes
the captured functional child, including its tensor computations; it does not
restore arbitrary Python hooks or side effects removed by export.

An existing plugin replaces the entire body, so its semantics and boundary
must match. For a plugin with schema `(Tensor h, Tensor bias) -> Tensor`:

~~~python
def bias_gelu(h, bias):
    return F.gelu(h + bias)

y = region.invoke(
    region.qdp(name="activation", op=torch.ops.plugins.bias_gelu.default),
    bias_gelu, h, bias,
)
~~~

A projection placed inside `bias_gelu` would also become part of the region and
would require a plugin implementing that larger computation.

Two conveniences lower to the same primitive:

~~~python
class Block(nn.Module):
    @region.decorate(
        region.generated(name="projection", provider="graphir")
    )
    def projection(self, x):
        return torch.relu(x @ self.weight + self.bias)

# Module form; the wrapper owns the original module, without copying weights.
self.attention = region.wrap(
    region.in_torch(name="attention"), Attention(...)
)
~~~

Decoration preserves method binding. Wrapping registers the body as a child
module. Both create region occurrences when invoked. Direct `invoke` remains
available to give two calls to the same module different policies.

Names are diagnostic labels, qualified by module path when available.
Discovery assigns occurrence IDs such as `block.projection#0` and
`block.projection#1` from graph traversal. Occurrences can have different
policies even when they share a body. Names and ordinals are not cache keys.

Specifications contain only stable policy. Provider objects, tuning budgets,
validation settings, and dataloaders belong to the compile call; section 5
shows that configuration. A generation `profile` selects provider settings,
not a TensorRT min/opt/max shape profile.

## 2. Capturing and discovering the boundary

The frontend adapter binds positional arguments, keyword arguments, and
defaults, then records their pytree structure. It presents a flat callable to
the HOP, reconstructs the original call inside it, and reconstructs the result
outside it. Tensor leaves are operands; supported Python scalar values are
specialized constants with guards. Outputs are Tensor pytrees.

For modules, the adapter exposes parameters and buffers before HOP capture
and invokes the body with explicit state, using a functional module call.
A bound-method adapter exposes the selected method as that functional call's
`forward`; it does not assume the method was already `forward`. Free functions
can take their Tensor state explicitly. Captured closure state requires the
same explicit binding or a capture error. This avoids relying on non-strict
export to discover hidden module state successfully.

For the prototype, the carrier can be PyTorch's experimental
[`hints_wrapper`](https://github.com/pytorch/pytorch/blob/main/torch/_higher_order_ops/hints_wrap.py).
It accepts flat operands and primitive hints, and currently rejects nonempty
body kwargs. The adapter therefore owns argument flattening and result
reconstruction. The public API hides this carrier; a supported upstream or
Torch-TensorRT-owned HOP can replace it.

Conceptually, capture produces:

~~~text
region_hop(
    body=child_graph,
    operands=(...),
    hints={"torch_tensorrt.region_schema": 1,
           "name": "attention", "route": "torch", ...},
)
~~~

Consider a callable attention body returning both context and probabilities:

~~~python
def attention(self, h, mask):
    q, k, v = h @ self.wq, h @ self.wk, h @ self.wv
    probs = ((q @ k.transpose(-2, -1)) * self.scale + mask).softmax(-1)
    return probs @ v, probs

def forward(self, x, mask):
    h = self.pre(x)
    ctx, probs = region.invoke(
        region.in_torch(name="attention"), self.attention, h, mask,
    )
    return self.post(ctx) + h, probs
~~~

After state normalization, the graph is structurally:

~~~text
parent(x, mask, Wpre, Wpost, Wk, Wq, Wv):
    h = pre(x, Wpre)
    ctx, probs = region_hop(attention_body, (h, mask, Wk, Wq, Wv), policy)
    out = post(ctx, Wpost) + h
    return out, probs

attention_body(h, mask, Wk, Wq, Wv):
    q, k, v = h @ Wq, h @ Wk, h @ Wv
    probs = softmax((q @ transpose(k)) * scale + mask)
    return probs @ v, probs
~~~

Here `scale` is a guarded static attribute. `h` and `mask` are activation
inputs; `Wq/Wk/Wv` are explicit read-only state. The two returned tensors are
the outputs. The residual use of `h` bypasses the region and does not make
`h` a region output.

Discovery reads this structure directly:

~~~python
# Conceptual compiler pass; helper names are proposed.
for node in parent.graph.nodes:
    if not is_region_hop(node):  # exact operator + versioned region hints
        continue
    spec, child, capture_binding = decode_region_hop(parent, node)
    abi = canonicalize_region_abi(parent, node, child, capture_binding)
    validate_region_contract(child, abi, spec)
    plans.append(RegionPlan(
        id=assign_occurrence_id(node, spec),
        spec=spec,
        reference=private_clone(child),
        abi=abi,
    ))
~~~

The pass validates the child reference, operand/placeholder correspondence,
output structure, schema version, and policy. Ordinary `hints_wrapper` nodes
without the region schema are not regions. Unsupported nesting and control-flow
containers fail explicitly. Source locations help diagnostics only.

The canonical ABI orders Tensor leaves by the bound public call's signature and
pytree traversal, excluding the receiver, followed by state slots in stable
qualified-name order. Static scalar attributes have separate bindings. Capture
can reorder, deduplicate, or omit operands, so the frontend must preserve a
source-slot mapping in the region manifest. For example, a body `f(x, y) = y - x`
may capture operands as
`(y, x)`; a QDP expecting `(x, y)` must still receive that order. Bind through
the recorded mapping, never raw HOP operand order or placeholder-name guesses.
Missing or ambiguous mappings are capture errors. A carrier/mode that cannot
supply this provenance requires an owned HOP adapter or remains unsupported.

`canonicalize_region_abi` uses that mapping to normalize child placeholders and
operands together, lift residual child-owned Tensor state, and record constants,
input/output pytrees, and shape/dtype/layout/device contracts. Repeated or unused
public arguments retain explicit logical bindings even if the captured child
does not need separate slots; rebuilding its signature may add unused
placeholders. Output count/order is checked against the adapter's TreeSpec.
This full ABI governs reference replay, plugin binding, and code generation;
activation samples alone are not the full ABI.

Support both strict and non-strict export where the carrier is proven to work;
retain the tracer's existing non-strict default. Normalize residual child
`get_attr` state rather than assuming export always lifted it. This pass cannot
repair a capture-time failure: bound methods, closures, kwargs, pytrees, and
state capture must be tested in each supported mode/version. Unsupported
combinations raise an actionable capture error, with an explicit strict-export
alternative where available; they must never silently erase the boundary.

## 3. Placement in the Torch-TensorRT pipeline

Region handling extends the current Dynamo/export path:

~~~text
torch_tensorrt.compile(model, region_options=..., region_data=...)
    -> owned export: invoke becomes a region HOP
    -> validate region HOPs
    -> region-aware pre-export lowering
    -> run_decompositions(): normalize parent and child graphs
    -> ep.module() + existing state/buffer lifting
    -> discover again + canonicalize ABI + build immutable reference plans
    -> collect boundary samples when needed
    -> resolve: PyTorch child | existing QDP | generated QDP | fallback
    -> existing post_lowering(), preserving region leaves
    -> support analysis and existing partitioners
    -> audit region ownership
    -> TensorRT engine construction + confirm owning engines
~~~

Discovery is rebuilt after decomposition because it returns a new
ExportedProgram; old FX node references are invalid. Pre-export lowering must
visit region children as well as the parent. Early validation runs before
autocast calibration; rule-based autocast with regions is rejected initially
until calibration and rewriting can visit child graphs.

The public compile entry owns a session across capture and resolution. It
removes provider/data objects from tracing kwargs, preserves real input tensors
before input preparation, and forwards both positional and keyword input
contracts. Resolution runs before CPU offload and before converter-support
analysis.

| Component | Responsibility / existing insertion point |
| --- | --- |
| `region/` (new) | Specifications, invoke/decorate/wrap, HOP and pytree adapter |
| [Public compile entry](py/torch_tensorrt/_compile.py) and [tracer](py/torch_tensorrt/dynamo/_tracer.py) | Session, options, capture compatibility, input provenance |
| [Lowering](py/torch_tensorrt/dynamo/lowering/passes/_aten_lowering_pass.py) | Recursive child normalization and preservation of boundaries |
| `dynamo/regions/` (new) | Discovery, ABI, reference/sample ownership, validation, resolution, placement manifest |
| [Dynamo compiler](py/torch_tensorrt/dynamo/_compiler.py) and [partitioners](py/torch_tensorrt/dynamo/partitioning) | Support rules, size exemptions, ownership/engine audits |
| torch-bear / GraphIR provider | Import child FX and produce a deployable kernel artifact or a rejection |
| [QDP adapter](py/torch_tensorrt/kernels/_ops.py) | Register selected operator, fake behavior, launch implementation, and converter |

## 4. Consuming regions before partitioning

Each occurrence resolves to one of these representations:

| Resolution | Graph transformation | Partitioning behavior |
| --- | --- | --- |
| `in_torch` or `on_failure="torch"` | HOP becomes one opaque `call_module` of its child | Must stay in PyTorch |
| Successful `qdp` / `generated` | HOP becomes one registered `call_function(op)` | Must enter TensorRT |
| `on_failure="partition"` | Inline a fresh reference child into the parent | Normal per-operation support rules |
| `on_failure="error"` | Raise with occurrence and failed stage | Compilation stops |

For an execute-in-Torch region, both fast and global support testers mark the
opaque call unsupported regardless of converters for its internal operations.
Hierarchical/resource partitioning must preserve that decision too. The
support summary and early-return logic must count these calls as unsupported,
and both compiler paths must skip their TensorRT conversion. The splitter
wires their arguments and results between partitions:

~~~text
TensorRT: pre(x) -> h
                     +----> PyTorch: attention(h, mask, weights) -> ctx, probs
                     |
                     +----> TensorRT: post(ctx) + h -> out
                                                        return out, probs
~~~

The region interior cannot split. An enclosing PyTorch partition may also
contain neighboring unsupported operations; a dedicated runtime partition per
region is unnecessary. CUDA tensors remain on CUDA across the boundary.

For a QDP replacement, bind the canonical ABI to the exact registered overload,
check schema/fake outputs and effects, validate the plugin, then replace the
HOP. Flat plugin outputs are reconstructed using the original output TreeSpec.
The attention example requires a plugin implementing its projections and
returning both outputs; a plugin accepting precomputed Q/K/V is a different
boundary.

Fusion and generated-kernel registration happen **before partitioning**. At
that point the compiler has one complete child to replace, and the converter is
visible during support analysis. Post-partition fusion would require undoing
splits, rebuilding boundaries, and repeating partition decisions. The
post-partition region pass is an audit, not a fusion pass.

Four compiler rules make placement reliable:

- Keep a compile-local `RegionPlacementManifest` with occurrence ID, expected
  leaf kind/target, and required route. Rebind it after rewrites and graph
  splitting; do not rely on stale node pointers or metadata alone.
- Treat resolved leaves as rewrite boundaries, including constant-fold
  exclusions. Validate the manifest after rewriting passes so a region cannot
  disappear, duplicate, or change route unnoticed.
- Give explicit QDP/generated nodes a node-local `min_block_size` exemption,
  including the compiler's early small-graph return and every partitioner's
  filter. Supported neighbors in the same component may share that exemption.
  Other calls to the same operator retain ordinary size rules.
- Audit that each PyTorch region is outside accelerated children and each QDP
  region belongs to exactly one accelerated child. Save that ownership before
  FX children become runtime modules, then confirm the owning engines exist.

Conflicting settings fail explicitly: PyTorch islands cannot satisfy
`require_full_compilation=True` or engine-only output; an explicitly excluded
QDP cannot also be required in TensorRT. Initially, force-PyTorch routes also
require `offload_module_to_cpu=False`, so their lifted state remains available
on its declared device.

Fallback covers region-attributable generation, ABI, numerical, and isolated
plugin-build failures before partitioning. Malformed capture and invalid shared
sample data are compilation errors. A later failure building a merged TensorRT
partition follows the entry point's existing build-failure policy; per-region
reconstruction and retry are deferred. Every fallback is reported.

## 5. From full-model data to region I/O

Compile-time configuration stays outside the model:

~~~python
def adapt_batch(batch):
    features, mask, _labels = batch
    return region.ModelCall(args=(features,), kwargs={"mask": mask})

compiled = torch_tensorrt.compile(
    model.eval(),
    inputs=compile_inputs,
    offload_module_to_cpu=False,
    region_options=region.RegionOptions(
        providers={"graphir": region.GraphIRProvider()},
        profiles={"default": region.KernelProfile(
            max_candidates=16, atol=1e-4, rtol=1e-3,
        )},
    ),
    region_data=region.RegionData(
        tuning_loader_factory=make_tuning_loader,
        validation_loader_factory=make_validation_loader,
        batch_adapter=adapt_batch,
        max_tuning_batches=16,
        max_validation_batches=8,
        max_capture_bytes=2 << 30,
    ),
)
~~~

The adapter explicitly supplies the full-model call, including kwargs where
needed. Each factory returns a fresh iterable; tuning is optional for providers
that do not search. Numerical tolerances are explicit acceptance settings and
must suit the model and dtype. QDP acceptance uses the default profile's
validation settings; generated regions select their named profile. Missing
provider or profile keys are configuration errors.

Torch-TensorRT retains an unresolved reference copy of the normalized parent.
For sample execution, each HOP becomes a uniquely named child call with the
same operands and outputs. A recorder around those calls performs one model
execution per batch and captures all requested occurrences:

~~~text
loader batch -> adapt_batch -> full-model reference
                                  |
                       embedding / preceding layers
                                  |
                        region pre-hook: operands
                        reference child: compute
                        region post-hook: outputs
                                  |
                           remaining layers

attention#0 sample:
    activations = (h, mask)
    state       = references to shared Wq/Wk/Wv snapshot
    expected    = (ctx, probs)
~~~

Validate the full-model input pytree, dtype, device, and export guards before
execution. Run the reference as exported under inference mode; verify eval mode
before capture. Snapshot activation inputs before the child and outputs
immediately afterward, preserving layout. Keep one case per occurrence and
batch; do not concatenate tensors along an assumed batch dimension.

Immutable parameters/buffers are recorded once and referenced by ABI slot.
Check that state is unchanged across collection and validation. Enforce one
memory budget across retained samples and state. Each standalone child is then
replayed with its full ABI and compared with the recorded parent result;
a mismatch is an extraction error, not a kernel-generation failure.

Real compile input tensors may seed a validation case when they form a complete
valid model call. Shape-only `torch_tensorrt.Input` specifications do not supply
semantic correctness data. Missing required validation data is reported before
replacement and follows the region's failure policy.

## 6. Generator contract and acceptance

The generator receives the computation and its execution contract, not Python
source recovered from the model:

~~~python
# Conceptual request: concrete type definitions are implementation details.
RegionKernelRequest(
    region_id=plan.id,
    reference_graph=clone(plan.reference),
    codegen_graph=clone(plan.reference),
    abi=plan.abi,                 # ordered slots and input/output TreeSpecs
    contracts=plan.contracts,     # shapes, dtypes, layouts, guards
    read_only_state=provider_state_snapshot(plan.state),
    tuning_cases=provider_tuning_snapshot(plan.tuning_cases),
    target=target,
    options=generation_profile,
)
~~~

Torch-TensorRT keeps its authoritative reference private. Providers may execute
or transform disposable clones; they cannot mutate the parent or decide their
own acceptance. State and tuning tensors are worker-owned copies, with the
authoritative state version retained and checked by the compiler. Providers
receive designated tuning cases. Held-out validation inputs and expected
outputs remain with the compiler, even though they use the same internal
sample representation. The two corpora must be separate; callers supply
distinct examples, and exact duplicate cases are rejected.

The direct provider imports functional FX through torch-bear, lowers through
GraphIR, and returns either a kernel artifact or a structured rejection.
Existing FX import/runtime compilation does not yet establish a standalone
GraphIR-to-QDP bridge.

| Artifact contract | Required contents |
| --- | --- |
| Code | Exact code bytes, format, entry symbol, target architecture |
| ABI and outputs | Argument order, Tensor bindings/static attributes, output shape/dtype/layout and allocation |
| Launch | Grid/block dimensions, shared memory, workspace and stream requirements |
| Validity | Supported shapes, layouts, alignment, effects, and target constraints |
| Identity | Digest, provider/toolchain versions, compilation options |

The initial generated path requires one externally launchable GPU kernel,
without hidden Python, ATen, FFI, or host-runtime execution. TensorRT supplies
output buffers. The physical argument order must match QDP's launch ABI or
pass an explicit validated adaptation.

The existing `torch_tensorrt.kernels.ptx_op` accepts precompiled PTX. Integration
still needs GraphIR to export PTX with this manifest, or a new adapter for its
native artifact. A runtime callable or CUBIN alone is insufficient.

Before committing a replacement, Torch-TensorRT checks the contract, evaluates
the actual candidate against the reference, and builds/runs an isolated
one-operator TensorRT engine. Comparing the reference with an eager fallback
that calls that same reference does not validate the kernel. Numerical policy
covers tolerances and NaN/Inf behavior; finite samples provide evidence, not a
proof over all possible inputs.

Test candidates in bounded worker processes. Register only the selected
artifact in the coordinator, using thread-safe, idempotent content-addressed
names and exact schema/artifact matching. Registrations are process-global and
cannot be assumed reversible.

An optional agent provider uses the same contract and compiler-owned evaluator.
Produce the direct GraphIR baseline first; an agent timeout or lack of
improvement keeps that baseline. An agent is a compile-time search component.

## 7. Diagnostics, persistence, and delivery

Each occurrence reports its source, requested/resolved route, boundary summary,
validation result, fallback reason, artifact/cache identity, and final
partition/engine owner. Dry-run performs structural analysis on a clone and
predicts placement; it does not iterate data, invoke providers, register
operators, or build engines. Untested generation and numerical results remain
unknown.

Cache identity includes normalized graph, ABI/contracts, target, provider
settings, and compatibility versions. Include state values only when explicitly
specialized; ordinary immutable weights remain Tensor bindings. Samples never
enter artifact caches or logs. Inference uses engines, required plugin runtime
artifacts, and any PyTorch children; it has no generator or agent dependency.
Packaging and fresh-process reload are acceptance requirements.

Deliver the feature in three slices:

1. Callable capture, ABI normalization, atomic PyTorch placement, existing QDP
   replacement, samples, and placement audits.
2. Strict native GraphIR export, QDP artifact adaptation, validation, and cache.
3. Optional agent search behind the same provider interface.

The initial scope is functional eval-mode inference with static dense CUDA
Tensor boundaries, static scalar inputs, and Tensor output pytrees. Provider
capability checks may narrow that scope; the first GraphIR demonstration should
be a small matmul/bias/activation region. Multi-output attention illustrates
region structure, not a promise of initial GraphIR coverage.

Reject input/state mutation, escaping aliases, RNG effects, graph breaks,
nested regions, and unsupported control flow. A future stateful algorithm must
return fresh state explicitly; opaque KV-cache mutation is not supported.
Dynamic shapes, refit/weight streaming, cross-compilation, and rule-based
autocast require separate compatibility work. The `torch.compile` backend
additionally needs capture activation before Dynamo runs, followed by discovery
and resolution after AOT normalization; the first delivery targets the owned
`torch_tensorrt.compile` export path.

Before enabling the API, verify strict/non-strict HOP capture and decomposition,
kwargs/pytrees and lifted state, repeated calls with different policies,
TRT/PyTorch/TRT placement under every partitioner, size exemptions and rewrite
barriers, sample replay, actual plugin numerics, fallback, and fresh-process
reload. The remaining integration decisions are the supported HOP/version
matrix and GraphIR's deployable artifact format.
