.. _lowering_passes_catalog:

Built-in Dynamo Lowering Passes
================================

This page catalogues every built-in lowering pass in
``torch_tensorrt.dynamo.lowering.passes``. Passes run sequentially on the FX graph
before partitioning. The ordering is fixed by the pass manager in
``pass_manager.py``; use the ``@_aten_lowering_pass(index=N)`` decorator to insert
custom passes at a specific position.

For a guide on writing new passes see :ref:`writing_dynamo_aten_lowering_passes`.

----

Pass Registry and Custom Passes
---------------------------------

Every pass is registered via the ``@_aten_lowering_pass`` decorator:

.. code-block:: python

    from torch_tensorrt.dynamo.lowering.passes import _aten_lowering_pass

    @_aten_lowering_pass(index=0)   # insert at the front
    def my_custom_pass(
        gm: torch.fx.GraphModule,
        settings: CompilationSettings,
    ) -> torch.fx.GraphModule:
        # modify gm.graph in-place
        gm.graph.lint()
        gm.recompile()
        return gm

Omitting ``index`` appends the pass to the end of the list. After structural changes
always call ``gm.graph.lint()`` and ``gm.recompile()``, or use the helper:

.. code-block:: python

    from torch_tensorrt.dynamo.lowering.passes import clean_up_graph_after_modifications
    clean_up_graph_after_modifications(gm)

----

Built-in Passes
----------------

repair_input_aliasing
^^^^^^^^^^^^^^^^^^^^^

**File**: ``repair_input_aliasing.py``

Inserts ``clone`` nodes immediately ahead of every input placeholder before the
lowering pipeline runs. This prevents mutation/aliasing bugs that can occur during
tracing when a downstream pass reads a placeholder that an earlier pass has silently
modified in-place (`pytorch#108079 <https://github.com/pytorch/pytorch/issues/108079>`_).

This pass is always paired with ``remove_input_alias_fixing_clones`` which strips the
clones again after the pipeline that required them has finished.

remove_assert_nodes
^^^^^^^^^^^^^^^^^^^^^

**File**: ``remove_assert_nodes.py``

Removes ``torch.ops.aten._assert_scalar`` and ``torch.ops.aten._assert_tensor_metadata``
nodes. These guard nodes are valid Python-side contracts but have no equivalent in TRT
and would cause conversion to fail if left in the graph.

remove_detach
^^^^^^^^^^^^^

**File**: ``remove_detach.py``

Replaces ``aten.detach`` nodes with their input. ``detach`` is a no-op at inference
time since TensorRT has no autograd concept; keeping it in the graph would require a
converter that does nothing.

constant_folding
^^^^^^^^^^^^^^^^

**File**: ``constant_folding.py``

Evaluates all subexpressions whose inputs are compile-time constants and replaces them
with ``get_attr`` nodes holding the folded result. Adapted from PyTorch Inductor's
``freezing.py``.

Benefits:

* Reduces engine build time — fewer nodes for TRT to optimize.
* Can eliminate entire branches (e.g., shape computations on static inputs).
* Converts intermediate tensors to registered parameters, allowing them to be treated
  as frozen weights in the TRT engine.

remove_num_users_is_0_nodes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**File**: ``remove_num_users_is_0_nodes.py``

Dead-code elimination: removes any ``call_function`` or ``call_method`` node whose
output has zero consumers (i.e., ``len(node.users) == 0``). Runs after
``constant_folding`` which may produce dead nodes as a side effect.

remove_input_alias_fixing_clones
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**File**: ``remove_input_alias_fixing_clones.py``

The counterpart of ``repair_input_aliasing``. Removes the temporary ``clone`` nodes
once the passes that needed aliasing protection have completed. The pair ensures the
graph presented to the partitioner and converter is clone-free.

repair_input_as_output
^^^^^^^^^^^^^^^^^^^^^^

**File**: ``repair_input_as_output.py``

TensorRT does not allow a network's input tensor to be directly returned as an output —
doing so would require TRT to copy the input, which it refuses. This pass detects any
placeholder node that appears in the graph's output tuple and inserts an
``aten.clone`` (identity copy) so that the output is a distinct tensor.

fuse_distributed_ops
^^^^^^^^^^^^^^^^^^^^^

**File**: ``fuse_distributed_ops.py``

Fuses distributed communication pairs into single atomic operations:

* ``all_gather_into_tensor`` + ``wait_tensor`` → ``tensorrt_fused_nccl_all_gather_op``
* ``reduce_scatter_tensor`` + ``wait_tensor`` → ``tensorrt_fused_nccl_reduce_scatter_op``

The fused forms expose the full gather/scatter as a single node that a single TRT plugin
can implement, avoiding a TRT↔PyTorch transition mid-communication. Only active when
``use_distributed_mode_trace=True``.

fuse_prims_broadcast
^^^^^^^^^^^^^^^^^^^^^

**File**: ``fuse_prims_broadcast.py``

Rewrites ``prims.sum`` + ``prims.broadcast_in_dim`` patterns into their ATen
equivalents (``aten.sum`` with ``keepdim=True``). The ``prims`` namespace is the
low-level decomposition layer below Core ATen; most converters target ATen, so this
pass lifts prims patterns back up to ATen for better converter coverage.

replace_max_pool_with_indices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**File**: ``replace_max_pool_with_indices.py``

Replaces ``aten.max_pool{1,2,3}d_with_indices`` + ``aten.getitem[0]`` sequences with
the simpler ``aten.max_pool{1,2,3}d`` variants when only the values (not the indices)
are consumed. The ``*_with_indices`` variants are harder for TRT to optimize; this
simplification enables the direct ``INetworkDefinition::addPoolingNd`` path.

rule_based_autocast
^^^^^^^^^^^^^^^^^^^^

**File**: ``rule_based_autocast.py``

When ``enable_autocast=True`` in ``CompilationSettings``, this pass inserts precision
cast nodes into the graph based on a set of rules:

1. **Output range check** — nodes whose outputs exceed ``autocast_max_output_threshold``
   are kept in FP32 (guarded against overflow).
2. **Reduction depth check** — nodes with reduction depth exceeding
   ``autocast_max_depth_of_reduction`` are kept in FP32 (guarded against accumulated
   error).
3. **Name exclusions** — nodes matching any pattern in ``autocast_excluded_nodes`` are
   kept in FP32.
4. **Op exclusions** — nodes with targets in ``autocast_excluded_ops`` are kept in FP32.
5. **Calibration** — if ``autocast_calibration_dataloader`` is provided, actual output
   ranges are measured by running the model on calibration data before applying the
   threshold check.

The pass inserts explicit ``aten._to_copy`` cast nodes around qualified operations to
reduce them to ``autocast_low_precision_type`` (FP16 or BF16).

remove_sym_nodes
^^^^^^^^^^^^^^^^^

**File**: ``remove_sym_nodes.py``

Removes ``sym_int`` placeholder nodes that ``torch.compile`` with ``dynamic=True``
inserts to represent symbolic integer values. These nodes carry shape information for
the tracing engine but have no runtime meaning for TRT. The pass replaces references
to symbolic integers with ``aten.sym_size`` queries on the concrete tensors, preserving
shape information while eliminating nodes TRT cannot process.

complex_graph_rewrite
^^^^^^^^^^^^^^^^^^^^^^

**File**: ``complex_graph_rewrite.py``

TensorRT has no complex-number dtype support. This pass detects subgraphs that operate
on ``complex64``/``complex128`` tensors and rewrites them to equivalent real arithmetic:

* **Input placeholders** with complex dtype are replaced by float placeholders with an
  appended trailing ``2`` dimension (``view_as_real`` layout).
* **``get_attr`` buffers** with complex dtype are unpacked into a float buffer of shape
  ``(*original_shape, 2)``.
* **``aten.mul``** between complex-typed nodes is rewritten to
  ``(ac - bd) + (ad + bc)i`` real arithmetic.
* **``aten.view_as_real``** and **``aten.view_as_complex``** nodes are erased (they
  become identity operations after the rewrite).

See :ref:`complex_tensor_lowering` (MEMORY.md) for invariants and limitations.
