.. _inplace_operations:

In-Place Operation Support
============================

This document describes the design for native (non-plugin) in-place operator
support in Torch-TensorRT. The motivating workload is streaming inference with a
key/value cache (e.g. ZoomASR, autoregressive LLM decoding) where the cache is
updated each step and the round-trip copy between PyTorch input buffer and
TensorRT output buffer dominates per-step cost.

Plugin-side aliasing (custom ops via
``PreviewFeature.ALIASED_PLUGIN_IO_10_03``) is intentionally out of scope here.
This design covers only built-in TensorRT operators and the runtime path for
declaring that an engine output shares its buffer with one of its inputs.

Implementation Status
^^^^^^^^^^^^^^^^^^^^^^

* **Implemented and verified end-to-end (C++ runtime path)**

  * The ``aten.slice_scatter.default`` decomposition is disabled; a converter
    handles it directly, emitting ``IKVCacheUpdateLayer`` when the cache is a
    direct network input with a 4-D static shape and write on dim 2,
    otherwise falling back to a scatter sequence.
  * ``aliased_io`` (mapping ``output_binding -> (input_binding, kind)``) is
    plumbed from the converter through ``TRTInterpreterResult``,
    ``SerializedInterpreterResult``, the Python wrapper, and the serialized
    engine blob (new ``ALIASED_IO_IDX`` at ABI v10).
  * The C++ runtime (``execute_engine``) honors the map: aliased outputs
    skip ``at::empty`` allocation, bind to the source input's ``data_ptr``,
    and are filtered from the user-facing return tuple. Pre-allocated
    outputs are disabled entirely when the engine has aliased I/O: an
    aliased output reuses the caller's input storage (there is no
    allocation to amortize), and caching would pin a caller-owned tensor
    across calls. Outputs are allocated fresh each call, and a warning is
    logged if ``use_pre_allocated_outputs`` was requested on an aliased
    engine.
  * ``TRTEngine`` constructor reconciles its build-time map against
    ``ICudaEngine::getAliasedInputTensor`` so the TRT API is the source of
    truth even for engines built outside Torch-TensorRT.
  * Streaming use case (``user passes the same cache each step``) works
    end-to-end: identity and ``data_ptr`` of the user's cache tensor are
    preserved across repeated calls; the engine writes in place.

* **Also implemented**: the ``BUFFER`` / ``BUFFER_MUTATION`` flow. A new
  ``lift_mutated_buffers`` pre-compile pass detects the trailing
  ``aten.copy_(get_attr, value)`` that ``ExportedProgram.module()``
  generates for a BUFFER_MUTATION, converts the ``get_attr`` to a
  ``placeholder``, and removes the trailing ``copy_``. The buffer becomes
  an engine input binding so the KV-cache fast path can fire. The compiled
  module is wrapped in ``BufferThreadingModule``, which owns the buffers
  as module state and threads them into each forward call. With aliased
  I/O the engine writes through the binding into the buffer's storage and
  the buffer state persists across calls — the user just calls
  ``module(x)`` without managing the cache.

Motivation
-----------

PyTorch's ``torch.export`` runs functionalization during decomposition. By the
time the FX graph reaches ``TRTInterpreter`` every in-place operation has been
rewritten to a functional equivalent followed by a ``copy_``::

    x.add_(y)             →   x_new = aten.add(x, y); x.copy_(x_new)
    cache.scatter_(...)   →   cache_new = aten.scatter(cache, ...);
                              cache.copy_(cache_new)

The compiled engine therefore produces a fresh output tensor and the wrapping
module copies it back into the user's input buffer. For workloads where the
mutated tensor is the dominant tensor (KV cache, ring buffers in streaming ASR)
this is a meaningful loss of bandwidth and an unnecessary allocation.

TensorRT exposes two relevant primitives:

* ``IKVCacheUpdateLayer`` — a built-in layer that performs a scatter into a
  static-shape cache and is automatically aliased to its cache input. The
  output binding shares device memory with the cache input.
* ``ICudaEngine.getAliasedInputTensor(output_name)`` — a runtime API that
  returns the input binding name an output is aliased with, or ``nullptr`` if
  the output is not aliased.

This design wires those primitives through the Torch-TensorRT pipeline so the
user can declare "this input is mutated" and the compiled engine will update it
in place, with no post-engine copy.

Background: TensorRT Primitives
--------------------------------

KVCacheUpdate Operator
^^^^^^^^^^^^^^^^^^^^^^^

The ``IKVCacheUpdateLayer`` performs ``output[i, :, writeIndices[i] + s, :] =
update[i, :, s, :]`` and aliases ``output`` to ``cache``. The cache layout is
``[batch, num_heads, s_max, head_dim]`` — the sequence axis (written to) is
axis 2. Inputs:

* ``cache`` — shape ``[b, num_heads, s_max, head_dim]``, network input, static
  ``s_max``.
* ``update`` — shape ``[b, num_heads, s, head_dim]`` with ``s ≤ s_max``.
* ``writeIndices`` — shape ``[b]``, ``int32`` or ``int64``, satisfying
  ``writeIndices[i] + s <= s_max``.

The output is the updated cache, which must be a network output and shares
memory with the cache input. K and V are independent layers. DLA is not
supported. The maximum sequence length must be static; dynamic ``s_max`` is not
permitted.

Aliased I/O Query API
^^^^^^^^^^^^^^^^^^^^^^

At runtime, ``engine->getAliasedInputTensor(out_name)`` returns the name of the
input binding aliased with the given output, or ``nullptr``. This is the source
of truth for the runtime: regardless of how aliasing was established (via
``IKVCacheUpdateLayer`` or any future API) the engine reports the
post-build wiring through this single call.

Design Overview
----------------

The work is layered into two tiers.

Tier A — KVCacheUpdate Fast Path
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A pattern-match lowering pass detects scatter-into-static-cache patterns and
rewrites them to a marker op ``torch_trt.kv_cache_update``. The converter for
that marker emits ``IKVCacheUpdateLayer``. Aliasing is automatic on the
TensorRT side; the runtime sees it through ``getAliasedInputTensor``.

Tier B — General Input/Output Aliasing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For mutated inputs that do not match the KV-cache pattern, the user supplies a
``mutated_inputs`` argument to :func:`torch_tensorrt.dynamo.compile`. The
converter records the pairing and the runtime treats the corresponding output
binding as aliased to the input binding — same device pointer, no fresh
allocation, and the input tensor is returned by identity.

Whether Tier B can also use a public TensorRT network-build API to declare
non-KV aliasing (without involving plugins) is an open question (see
:ref:`open_questions`). If TensorRT 10.x does not expose such an API, Tier B
collapses to Tier A and ships only KV-cache support; the rest of this design
still applies in that case.

User-Facing API
----------------

The ``compile()`` entry point gains one new keyword argument:

.. code-block:: python

    compiled = torch_tensorrt.dynamo.compile(
        exported_program,
        inputs=[cache_k, cache_v, x],
        mutated_inputs={"cache_k": "cache_k_out",
                        "cache_v": "cache_v_out"},
    )

    # cache_k and cache_v are mutated in place; out[2] is the real output.
    out = compiled(cache_k, cache_v, x)
    assert out[0] is cache_k
    assert out[1] is cache_v

``mutated_inputs`` maps an input binding name (or index) to the output binding
name (or index) that should alias it. When the compiled module is called, the
aliased output positions in the returned tuple contain the *same* ``at::Tensor``
that was passed in — same storage, same ``data_ptr()``, observably mutated.
This mirrors PyTorch's in-place op convention in which ``x.add_(y)`` returns
``x``.

Implementation Phases
----------------------

Lowering
^^^^^^^^^

A new post-lowering pass ``mark_aliased_outputs`` runs after
``remove_input_alias_fixing_clones``. For each entry in ``mutated_inputs``:

1. Resolve the input ``placeholder`` node and the corresponding output node.
2. Tag the output node with ``node.meta["aliased_input"] = "<input_name>"``.

The metadata travels through to ``TRTInterpreter`` and is the carrier for
aliasing intent.

A second lowering sub-pass detects KV-cache-shaped patterns among the
aliased-output nodes:

* Scatter or ``index_put`` into a tensor of shape
  ``[b, num_heads, s_max, head_dim]``.
* ``writeIndices`` is a ``[b]``-shaped integer tensor.
* ``s_max`` is static.

Matching nodes are replaced with a ``torch_trt.kv_cache_update`` marker so the
converter can emit ``IKVCacheUpdateLayer`` directly. Non-matching aliased
outputs fall through to Tier B.

Conversion
^^^^^^^^^^^

Two touch points in ``py/torch_tensorrt/dynamo/conversion/_TRTInterpreter.py``:

1. **New converter** for ``torch_trt.kv_cache_update`` which calls
   ``net.add_kv_cache_update(cache, update, write_indices)``. The cache input
   must already be added via ``net.add_input``; converters look up its
   ``ITensor`` handle via the conversion context.

2. **``output()`` (around line 837)** — when marking each output, check
   ``node.meta["aliased_input"]``. If present and the output came from
   ``IKVCacheUpdateLayer``, no extra call is needed (TRT aliases automatically).
   For Tier B (if the public API exists) call the corresponding TRT method
   before ``mark_output``.

The interpreter records the alias map on its result type:

.. code-block:: cpp

    struct TRTInterpreterResult {
        // ...existing fields
        std::unordered_map<std::string, std::string> aliased_io;  // out_name → in_name
    };

This map is the bridge from build-time intent to the runtime engine object.

C++ Runtime Changes
^^^^^^^^^^^^^^^^^^^^

All runtime work lives under ``core/runtime/``.

``TRTEngine`` (``core/runtime/TRTEngine.h``)
   Add one field::

       std::unordered_map<std::string, std::string> aliased_io;  // out → in

   Populated once at construction from two sources:

   1. The serialized engine metadata (see :ref:`serialization_format`).
   2. **Source-of-truth reconciliation:** for every output binding, query
      ``cuda_engine->getAliasedInputTensor(out_name)`` and merge the result
      into ``aliased_io``. This runs a single time in the ``TRTEngine``
      constructor, not per execution, so the reconstructed map is reused across
      all calls.

   For the ``kv_cache_update`` kind the two sources are redundant: the engine
   reports the same aliasing through ``getAliasedInputTensor``, so reconciliation
   alone can rebuild it — this is what lets engines built from external TensorRT
   plans (e.g. via ``IKVCacheUpdateLayer`` from a non-Torch-TRT build flow) load
   transparently even with an empty serialized map. The serialized metadata is
   therefore strictly required only for the ``user`` kind, which TRT has no API
   to report, and to preserve the ``kind`` tag. We still serialize the KV
   entries for completeness and self-description; dropping them for KV-only
   engines and relying entirely on reconciliation would be a valid future
   simplification.

``execute_engine`` (``core/runtime/execute_engine.cpp``)
   Three narrow changes:

   1. **Input binding** — record each contiguous input's ``data_ptr()``
      keyed by binding name into a local ``input_addrs`` map. The existing
      ``setTensorAddress`` call is unchanged.

   2. **Output binding (~line 188)** — branch:

      .. code-block:: cpp

          if (auto it = engine.aliased_io.find(out_name);
              it != engine.aliased_io.end()) {
              // Aliased output: bind the same device ptr as its input,
              // do NOT allocate, and return the input tensor by identity.
              void* aliased_ptr = input_addrs.at(it->second);
              ctx->setTensorAddress(out_name.c_str(), aliased_ptr);
              output_tensors.push_back(input_tensors_by_name.at(it->second));
          } else {
              auto out = at::empty(dims, options).contiguous();
              ctx->setTensorAddress(out_name.c_str(), out.data_ptr());
              output_tensors.push_back(out);
          }

   3. **Shape consistency check** — before binding, assert
      ``dims == input_tensor.sizes()`` for aliased pairs. A mismatch
      indicates compilation produced an output shape that differs from the
      input it claims to alias; abort with a clear error rather than silently
      corrupting memory.

Output allocator interaction
   The existing ``OutputAllocator`` path (used when
   ``requires_output_allocator=true``) is incompatible with aliasing by
   construction: aliasing requires the output's storage to match the input's,
   while the output allocator exists precisely because the output shape is not
   known ahead of time. ``TRTEngine`` construction validates that no binding
   appears in both ``aliased_io`` (as an output) and the
   ``requires_output_allocator`` set, throwing on construction if it does.

Stream and synchronization
   Unchanged. Aliased I/O is a pointer-identity trick, not a synchronization
   trick. The pre-/post-enqueue stream handling remains valid.

CUDA Graph capture
   Aliased I/O makes capture *more* deterministic — fewer allocations, stable
   addresses. The user-supplied input tensor's ``data_ptr()`` must be stable
   across replays; if the caller passes a different tensor each call, capture
   is invalidated, the same as today for non-aliased inputs.

Python Runtime Changes
^^^^^^^^^^^^^^^^^^^^^^^

The pure-Python runtime mirrors the C++ engine so a compiled artifact behaves
identically on either runtime. It honors aliased I/O natively — it does **not**
merely warn and fall back to fresh allocation. All work lives under
``py/torch_tensorrt/dynamo/runtime/_TRTEngine.py``.

``TRTEngine`` (``_TRTEngine.py``)
   Add one field::

       self.aliased_io: Dict[str, Tuple[str, str]]  # out_name → (in_name, kind)

   Populated once at setup from two sources, exactly like the C++ constructor:

   1. The serialized engine metadata (``ALIASED_IO_IDX``), decoded in
      ``_load_serialized_info`` (see :ref:`serialization_format`).
   2. **Source-of-truth reconciliation:** ``_reconcile_aliased_io`` queries
      ``cuda_engine.get_aliased_input_tensor(out_name)`` for every output and
      merges the result. This runs a single time at setup, not per execution.

   ``kind="user"`` entries are preserved (TRT has no API to report them); a
   ``kv_cache_update`` alias the engine reports but the serialized map lacks is
   discovered; a disagreement is resolved in favor of the engine. The pass also
   recomputes ``aliased_input_binding_names`` so the per-call input loop tests
   membership in O(1) (parallel to the C++ ``input_binding_infos`` ``is_aliased``
   flag).

``execute`` (``_execute_standard`` / ``setup_input_tensors`` / ``create_output_tensors``)
   Three narrow changes:

   1. **Input binding** — ``setup_input_tensors`` records each contiguous input's
      tensor keyed by binding name into ``self._bound_inputs_by_name``. For an
      aliased input it binds the user's tensor directly and skips the cudagraph
      persistent staging-buffer clone (staging would break the aliasing).

   2. **Output binding** — branch:

      .. code-block:: python

          alias = self.aliased_io.get(output_name)
          if alias is not None:
              # Aliased output: reuse the source input tensor by identity, do
              # NOT allocate, and bind the engine output to the same data_ptr.
              aliased_input = self._bound_inputs_by_name[alias[0]]
              outputs.append(aliased_input)
              self.context.set_tensor_address(output_name, aliased_input.data_ptr())
          else:
              out = torch.empty(self.output_shapes[idx], ...)
              outputs.append(out)
              self.context.set_tensor_address(output_name, out.data_ptr())

      (allocation/identity reuse lives in ``create_output_tensors``; the address
      binding and the cudagraph output-staging bypass live in
      ``_execute_standard``.)

   3. **Shape consistency check** — ``create_output_tensors`` asserts
      ``aliased_input.shape == output_shape`` before reusing the input's storage;
      a mismatch aborts rather than silently corrupting memory.

   Pre-allocated outputs are disabled entirely when ``aliased_io`` is non-empty
   (via the ``has_aliased_io`` argument to
   ``TorchTRTRuntimeStates.set_runtime_states`` and the cache-refresh guard); a
   warning is logged if ``use_pre_allocated_outputs`` was requested on an aliased
   engine.

Output allocator interaction
   As in C++, the dynamic output-allocator path is incompatible with aliasing.
   ``_reconcile_aliased_io`` raises ``RuntimeError`` if an aliased output is
   present while ``requires_output_allocator`` is set.

Stream and synchronization / CUDA Graph capture
   Unchanged and identical to the C++ runtime: aliased I/O is a pointer-identity
   trick, so the stream choreography and cudagraph record/replay paths behave the
   same as above, minus the aliased output-staging copy-back.

.. _serialization_format:

Serialization Format Update
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The C++ engine serialization format in ``core/runtime/runtime.h`` is a
positional ``std::vector<std::string>`` indexed by an enum, gated by a string
``ABI_VERSION``. The change is additive: one new index, one new field,
one ABI bump.

In ``core/runtime/runtime.h``:

.. code-block:: cpp

    const std::string ABI_VERSION = "10";  // bumped from "9"

    typedef enum {
        ABI_TARGET_IDX = 0,
        // ...existing entries unchanged...
        REQUIRES_NATIVE_MULTIDEVICE_IDX,
        ALIASED_IO_IDX,             // NEW
        SERIALIZATION_LEN,
    } SerializedInfoIndex;

    std::string serialize_aliased_io(
        const std::unordered_map<std::string, std::string>& aliased_io);
    std::unordered_map<std::string, std::string> deserialize_aliased_io(
        const std::string& s);

Encoding follows the same convention as ``serialize_bindings``: pairs joined
with a key/value delimiter, records joined with a record delimiter. No JSON,
no protobuf, no new dependencies. An empty map serializes to the empty
string.

In ``core/runtime/TRTEngine.cpp``:

* **Constructor (line 85)** delegates one extra deserialized argument to the
  primary constructor::

      deserialize_aliased_io(serialized_info[ALIASED_IO_IDX])

* **``serialize()`` (line 508)** writes one extra entry::

      serialized_info[ALIASED_IO_IDX] =
          serialize_aliased_io(this->aliased_io);

* **``__obj_flatten__()`` (line 484)** gains a corresponding tuple so
  Python introspection sees the new field::

      std::tuple("aliased_io", serialized_info[ALIASED_IO_IDX]),

* **``verify_serialization_fmt`` (line 471)** is unchanged. Its existing length
  check (``size() == SERIALIZATION_LEN``) plus the ABI version check
  collectively reject any pre-bump engine cleanly.

Compatibility
   Pre-bump engines (ABI ``"9"``) are rejected by ``verify_serialization_fmt``
   with the existing ABI mismatch error. Users recompile. This matches the
   behavior of all prior ABI bumps; the format is intentionally not
   forward/backward compatible across version changes.

   Engines with no aliased outputs serialize ``aliased_io`` to the empty
   string and take the existing allocate-fresh-output path at runtime. Zero
   behavioral change for unaffected users.

.. _buffer-style:

Buffer-Backed KV Cache
-----------------------

PyTorch's canonical pattern for streaming inference is to hold the cache as a
module buffer:

.. code-block:: python

    class StreamingKV(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("cache_k", torch.zeros(B, H, S_MAX, D))

        def forward(self, x_k):
            self.cache_k[:, :, t:t+1, :] = x_k
            return attention(self.cache_k, x_k)

``torch.export`` captures this cleanly: ``cache_k`` becomes a ``BUFFER``
input in ``graph_signature.input_specs`` and the slice-write becomes a
``BUFFER_MUTATION`` output spec.

``ExportedProgram.module()`` rewrites this into a ``GraphModule`` where the
buffer is read via ``get_attr`` and the mutation is emitted as a trailing
``aten.copy_(get_attr_buffer, slice_scatter_result)`` node. Left alone,
Torch-TensorRT's constant-folding pass would bake the buffer into the
engine as a weight; the slice-scatter would scatter against that constant
and produce a result the engine discards.

To preserve buffer state across calls and let the KV-cache aliasing fast
path fire, a pre-compile pass ``lift_mutated_buffers`` runs after
``ep.module()`` and before ``post_lowering``:

1. Scans for ``aten.copy_.default(get_attr, _)`` patterns — the marker for
   a BUFFER_MUTATION.
2. For each match, converts the ``get_attr`` to a ``placeholder``,
   redirects all uses, and erases the trailing ``copy_``.
3. Rebuilds the ``GraphModule`` with the default ``CodeGen`` so the
   ``forward`` signature reflects the new placeholder set (the original
   ``_PyTreeCodeGen`` would re-impose the original arity through a stored
   pytree spec).

The compiled result is wrapped in ``BufferThreadingModule``, which owns
the buffers as ``register_buffer`` state and threads them into the
underlying compiled module on each forward call. Combined with the
engine-level KV-cache aliasing, the engine writes directly into the
buffer's storage; the buffer is observably mutated and the next call
reads the updated state. The user-facing API is just ``module(x)``.

Both the user-passed-cache pattern (caller owns the cache) and the
buffer-backed pattern (module owns the cache) work and produce identical
results.

Side effects on non-KV buffer mutations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``lift_mutated_buffers`` keys off the generic ``aten.copy_(get_attr, _)``
marker, not off the KV-cache shape, so it lifts *every* mutated buffer — not
only KV caches. This changes how any buffer-mutating module is compiled: the
mutated buffer becomes an engine input binding and its trailing ``copy_`` is
dropped, whereas previously it was constant-folded into the engine.

For a mutation that also matches the KV-cache fast path this is the whole
point — the engine aliases the output back into the buffer's storage and the
mutation is observable across calls. For a mutation that does **not** match
the fast path (wrong rank/shape/dim, dynamic ``s_max``, a plain elementwise
``copy_``), the converter falls back to the scatter path and the output is
**not** aliased. The buffer is still lifted to an input and threaded through
``BufferThreadingModule``, so the mutated value is produced and carried
forward correctly, but without the in-place aliasing benefit — behavior is
equivalent to the pre-existing copy-back, just wired through a binding rather
than a folded constant. Ordinary functional ``.copy_`` residue that is *not*
a ``get_attr`` buffer mutation (e.g. the functionalization tail of an
in-place op on an activation) does not match the ``get_attr`` first-arg
pattern and is left untouched.

Constraints and Known Limitations
----------------------------------

Static cache shape (Tier A)
   ``IKVCacheUpdateLayer`` requires static ``s_max``. Streaming ASR with a
   fixed context window satisfies this; truly dynamic-length caches do not
   and fall through to Tier B.

Single-input aliasing
   Each output binding aliases at most one input binding. There is no design
   here for an output that aliases multiple inputs; that has no clear
   semantics anyway.

Tensor identity through PyTorch
   When ``execute_engine`` returns the aliased input tensor as one of its
   outputs, downstream code observes ``out is input`` as ``True``. Wrappers in
   ``_TorchTensorRTModule.forward()`` that touch outputs (e.g. an unsuspecting
   ``output.contiguous()``) must remain a no-op for already-contiguous
   tensors; this is the existing PyTorch contract and the design relies on
   it.

DLA
   ``IKVCacheUpdateLayer`` is not supported on DLA. ``mutated_inputs`` on a
   DLA target raises at compile time.

User responsibility for buffer reuse
   Buffers passed in via ``mutated_inputs`` are mutated by the engine. Users
   that need a pre-mutation snapshot must clone before calling. This is
   consistent with PyTorch in-place op semantics.

.. _open_questions:

Open Questions
---------------

Tier B network-build API
   ``IKVCacheUpdateLayer`` aliases its output automatically, which fully
   covers Tier A. Whether TensorRT 10.x exposes a public network-build API
   for declaring output↔input aliasing on arbitrary layers (without involving
   the plugin path) is an open question. If such an API does not exist, Tier
   B is deferred and the design ships Tier A only — the runtime, lowering,
   and serialization changes described above remain valid; only the second
   lowering sub-pass and its corresponding converter route are unbuilt.

ZoomASR fit
   The KVCacheUpdate constraints (static ``s_max``, K/V split, ``[b, d, s_max,
   h]`` layout) need confirmation against the ZoomASR cache layout. If the
   model uses a different memory layout (e.g. ``[b, s_max, num_heads, head_dim]``) a
   ``permute`` may be required, and the cost of that permute may exceed the
   savings from aliasing. A small benchmark on the actual model is the
   gating criterion before investing in pattern-matching.

Phased Rollout
---------------

1. **Phase 1 — Tier A only.** Lowering pass, converter for
   ``IKVCacheUpdateLayer``, C++ runtime aliased-output path,
   serialization-format bump. This is the smallest end-to-end surface that
   solves the streaming-ASR / KV-cache use case.

2. **Phase 2 — Tier B (general aliasing).** Only if the TensorRT public API
   supports non-KV aliasing. Reuses the same runtime code path; new converter
   route. No further runtime or serialization changes.

3. **Phase 3 — Auto-detection.** Pattern-match common in-place residue
   post-functionalization (e.g. ``copy_`` of a scattered tensor back into
   the original input) so users do not need to specify ``mutated_inputs``
   for the common case. Ergonomics-only; behavior identical to Phase 1/2.

Summary of Code Touch Points
-----------------------------

+-------------------------------------------+--------------------------------------------------------------------+
| File                                      | Change                                                             |
+===========================================+====================================================================+
| ``core/runtime/runtime.h``                | Bump ``ABI_VERSION`` to ``"10"``; add ``ALIASED_IO_IDX``;          |
|                                           | declare aliased-IO serialization helpers.                          |
+-------------------------------------------+--------------------------------------------------------------------+
| ``core/runtime/runtime.cpp``              | Implement ``serialize_aliased_io`` /                               |
|                                           | ``deserialize_aliased_io``.                                        |
+-------------------------------------------+--------------------------------------------------------------------+
| ``core/runtime/TRTEngine.h``              | Add ``aliased_io`` field; update constructor signatures.           |
+-------------------------------------------+--------------------------------------------------------------------+
| ``core/runtime/TRTEngine.cpp``            | Read/write ``ALIASED_IO_IDX``; populate from                       |
|                                           | ``getAliasedInputTensor`` post-deserialize; add to                 |
|                                           | ``__obj_flatten__``.                                               |
+-------------------------------------------+--------------------------------------------------------------------+
| ``core/runtime/execute_engine.cpp``       | Branch in output binding loop: skip ``at::empty`` for aliased      |
|                                           | outputs, bind input ``data_ptr``, return input tensor by identity. |
+-------------------------------------------+--------------------------------------------------------------------+
| ``py/.../lowering/passes/``               | New ``mark_aliased_outputs`` pass; KV-cache pattern matcher.       |
+-------------------------------------------+--------------------------------------------------------------------+
| ``py/.../conversion/_TRTInterpreter.py``  | New converter for ``torch_trt.kv_cache_update``; aliased-output    |
|                                           | handling in ``output()``; ``aliased_io`` on                        |
|                                           | ``TRTInterpreterResult``.                                          |
+-------------------------------------------+--------------------------------------------------------------------+
| ``py/.../dynamo/_compiler.py``            | New ``mutated_inputs`` argument; plumb to lowering pass and        |
|                                           | engine builder.                                                    |
+-------------------------------------------+--------------------------------------------------------------------+
