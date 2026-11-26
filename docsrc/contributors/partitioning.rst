.. _partitioning:

Partitioning Phase
====================

The partitioning phase splits the lowered ``torch.fx.Graph`` into subgraphs that will
run in TensorRT and subgraphs that will remain in PyTorch. Each TensorRT subgraph
becomes a separate compiled engine; the PyTorch subgraphs are kept as-is and stitched
together with the engines in the final module.

Dynamo Partitioning (Primary Path)
------------------------------------

Capability Partitioning
^^^^^^^^^^^^^^^^^^^^^^^^^

The fundamental question the partitioner answers is: *can this node be converted to
TensorRT?* A node is considered TensorRT-capable when **all** of the following hold:

1. A converter is registered for its ``target`` in ``DYNAMO_CONVERTERS``.
2. The converter's **capability validator** returns ``True`` for this specific node.
3. The node is not listed in ``torch_executed_ops``.
4. The node is not an impure op (unless it is a ``get_attr``).

Nodes that fail any of these checks fall back to PyTorch execution.

The Validator System
^^^^^^^^^^^^^^^^^^^^^

Every converter entry is a ``ConverterSupport`` dataclass:

.. code-block:: python

    @dataclass(frozen=True)
    class ConverterSupport:
        converter_implementation: ConverterImplSignature
        capability_validator: Callable[[Node, CompilationSettings], bool] = lambda node, settings: True
        supports_dynamic_shapes: bool = False
        requires_output_allocator: bool = False

The **``capability_validator``** is a callable ``(node, settings) -> bool`` that is
invoked at partition time — before compilation — against the live FX node. Validators
enable fine-grained routing that cannot be expressed in the converter's type signature
alone. Common uses:

* Restricting a converter to a subset of its overloads (e.g. only ``approximate="tanh"``
  GELU).
* Blocking a converter when a specific kwarg is present that the TRT implementation
  does not support.
* Deferring to PyTorch when the node has dynamic shapes and the converter has not
  been validated for them.

**Dynamic shape validation** is handled via built-in helpers in the registry:

.. code-block:: python

    # True if any input/output dimension is symbolic
    has_dynamic_shapes(node) -> bool

    # True if specific positional arguments have symbolic shapes
    has_dynamic_shapes_in_args(arg_positions=[0, 1]) -> Callable

    # Inverse — True if all checked positions are static
    has_static_shapes_in_args(arg_positions=[0, 1]) -> Callable

If ``CompilationSettings.assume_dynamic_shape_support = True``, all dynamic-shape
validator checks are skipped and every converter is treated as dynamic-shape capable.

The **``supports_dynamic_shapes``** flag on ``ConverterSupport`` is also checked during
the registry lookup: if the node has symbolic dimensions and the best matching converter
has ``supports_dynamic_shapes=False``, the node is routed to PyTorch.

The **``requires_output_allocator``** flag marks converters whose TRT implementations
produce data-dependent output shapes (e.g. ``nonzero``, ``unique``). The runtime will
use TensorRT's output allocator for these ops rather than pre-allocating fixed buffers.

Checking Converter Coverage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before or after partitioning you can inspect how many nodes in the graph are TRT-capable:

.. code-block:: python

    from torch_tensorrt.dynamo.partitioning import get_graph_converter_support

    n_supported, n_total = get_graph_converter_support(
        graph_module,
        torch_executed_ops=set(),
    )
    print(f"{n_supported}/{n_total} ops supported by TensorRT")

This calls ``is_node_supported()`` on every ``call_function`` node and prints a
per-operator breakdown.

Partitioning Strategies
^^^^^^^^^^^^^^^^^^^^^^^^^

Three strategies are available. All are in ``torch_tensorrt.dynamo.partitioning``.

**Adjacency Partitioning (default — ``fast_partition``)**

.. code-block:: python

    from torch_tensorrt.dynamo.partitioning import fast_partition

    partitioned_gm, support_overview = fast_partition(
        gm,
        min_block_size=5,
        torch_executed_ops=set(),
        require_full_compilation=False,
    )

Traverses the graph in topological order and greedily merges adjacent TRT-capable nodes
into the same subgraph. Uses ``FxNetAccFusionsFinder`` to keep operator fusions intact
across the boundaries. Fast at compile time; produces a near-optimal partition for the
vast majority of models.

**Global Partitioning (``global_partition``)**

.. code-block:: python

    from torch_tensorrt.dynamo.partitioning import global_partition

    partitioned_gm, support_overview = global_partition(
        gm,
        min_block_size=5,
        torch_executed_ops=set(),
        require_full_compilation=False,
    )

Uses PyTorch's ``CapabilityBasedPartitioner`` to compute the globally optimal partition
— the one with the fewest subgraphs. Useful when context-switch overhead between
PyTorch and TensorRT is significant and you need to guarantee the minimum number of
engine/PyTorch transitions. Slower at compile time than adjacency partitioning.

Both strategies respect:

* **``min_block_size``** — TRT subgraphs with fewer than this many operators are merged
  back into the adjacent PyTorch subgraph. Prevents tiny engines that are slower than
  just running in PyTorch.
* **``require_full_compilation``** — if ``True``, raises an error if any node cannot be
  placed in TensorRT.
* **``torch_executed_ops``** — set of ``torch.ops`` targets that are always forced to
  PyTorch regardless of converter availability.

**Hierarchical / Multi-backend Partitioning (``hierarchical_adjacency_partition``)**

.. code-block:: python

    from torch_tensorrt.dynamo.partitioning import hierarchical_adjacency_partition

    partitioned_gm, support_overview = hierarchical_adjacency_partition(
        gm,
        min_block_size=5,
        backend_support_map={
            "tensorrt": set(DYNAMO_CONVERTERS.keys()),
            "inductor": my_inductor_ops,
        },
        backend_priority=["tensorrt", "inductor"],
    )

Extends the adjacency partitioner to support multiple backends with a priority order.
Each node is assigned to the highest-priority backend whose capability validator
accepts it. Submodules are tagged with the backend name (e.g.
``_run_on_acc_tensorrt_0``, ``_run_on_acc_inductor_1``). Useful for research into
multi-backend compilation or when combining TensorRT with another compiled backend.

See the
`hierarchical_partitioner_example <https://github.com/pytorch/TensorRT/blob/main/examples/dynamo/hierarchical_partitioner_example.py>`_
for a full walkthrough.

Resource Partitioning
^^^^^^^^^^^^^^^^^^^^^^

Large models can exhaust CPU RAM during TensorRT engine building because TensorRT holds
multiple copies of the graph in memory during optimization. **Resource partitioning**
splits oversized TRT subgraphs into smaller pieces and builds them one after another, so
each build holds only part of the model.

It has two controls:

* ``cpu_memory_budget``, the host memory the compilation may use. The partitioner derives
  the size of each engine from it.
* ``max_partition_size``, the size of each engine itself. It is a finer control
  over the same split; see `Fixing the Engine Size`_ below.

.. code-block:: python

    from torch_tensorrt.dynamo.partitioning._resource_partitioner import resource_partition

    # After capability partitioning, split subgraphs that exceed memory budget
    gm = resource_partition(gm, cpu_memory_budget=8 * 1024**3)  # 8 GB budget

Or set it via ``CompilationSettings``:

.. code-block:: python

    trt_gm = torch_tensorrt.compile(
        model,
        arg_inputs=inputs,
        enable_resource_partitioning=True,
        cpu_memory_budget=8 * 1024**3,
    )

How the split is computed:

1. ``ResourcePartitioner`` reads the process's resident memory (RSS) via ``psutil``, or the
   available system memory when ``cpu_memory_budget`` is not set.
2. It estimates the size of each TRT subgraph as the bytes of the weights (``get_attr``
   tensors) its nodes read. A weight read by two subgraphs counts in both, because each
   engine stores its own copy.
3. The per-engine size is ``(cpu_memory_budget - RSS) // ENGINE_COMPILATION_MEMORY_USAGE_MULTIPLIER``,
   or ``available_memory // ENGINE_COMPILATION_MEMORY_USAGE_MULTIPLIER`` without a budget
   (default multiplier: 4, the assumption that building an engine takes about 4 times its
   weights in host memory).
4. If the whole model is within that size, nothing is split. A model more than
   ``MAX_NUM_OF_ENGINES`` (50) times that size fails with an error instead.
5. Otherwise, each larger subgraph is split from the front, in graph order. Nodes move into
   a new engine in steps; a step that would take the engine over the size is undone and
   retried with fewer nodes. A single node larger than the size gets an engine of its own,
   which is then larger than the size.

The splitter keeps **atomic subgraphs** (operator fusions) intact. It will not split a
fused ``conv → bn → relu`` pattern across a boundary: it moves the whole fusion, and the
producers it needs, into the earlier engine.

Empirically, resource partitioning has negligible runtime cost — TRT can optimise
within each piece nearly as well as across the full subgraph.

Fixing the Engine Size
^^^^^^^^^^^^^^^^^^^^^^^

``max_partition_size`` sets the per-engine size of step 3 directly, in bytes of
weights, instead of deriving it from ``cpu_memory_budget`` and the resident memory. Every
other step stays the same:

.. code-block:: python

    trt_gm = torch_tensorrt.compile(
        model,
        arg_inputs=inputs,
        max_partition_size=512 * 1024**2,  # 512 MiB of weights per engine
    )

Use it when:

* **The split must be the same on every run.** The size derived from the budget changes
  with the process's resident memory when partitioning starts, which depends on things
  unrelated to the model: what the caller still holds, the allocator, and weights that are
  memory-mapped and not yet read. With the same budget, a 3.6-billion-parameter robot
  policy split into 27 engines in one run and 13 in another. With a fixed size, the split
  depends only on the model and the size.
* **Programs need different sizes.** Each ``compile`` call takes its own size, so an
  application that compiles several programs can split a large one finely and keep a
  small one in a single engine.
* **You want to choose the trade-off yourself.** A smaller size lowers the memory peak of
  each build, at the cost of more engines.

How it combines with the budget:

* Setting it turns resource partitioning on, so ``enable_resource_partitioning`` need not
  be set.
* It replaces ``cpu_memory_budget``, so setting both raises a ``ValueError``. The resident
  memory is not read.
* Internally, the partitioner treats the size as a budget of
  ``4 × max_partition_size``, so the checks of step 4 follow from it: a model
  within the size stays one engine, and a size so small that the model is more than 50
  times it fails with an error that names ``max_partition_size``.

What it does not bound:

* **The memory of a build.** The size counts an engine's weights. The memory TensorRT
  takes to build that engine is a multiple of it, which the factor of 4 only estimates;
  builds have been measured at 3 to 10 times their weights. Check the peak of a first
  compile, and lower the size if it is too high. Each split logs its engines' weights at
  ``INFO`` level.
* **Layers larger than the size.** A single layer larger than the size still gets an
  engine of its own, larger than the size. Lowering the size below the largest layer
  doesn't make that layer's engine any smaller.

Recomputing Input-Only Values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every split puts engine boundaries inside the model: the adjacency and global partitioners
split around operators that run in PyTorch, and the resource partitioner splits by size. A
model computes the values that depend only on its inputs, such as an attention mask or rotary
``cos``/``sin``, before its first layer, so they land in the first engine. Instead of passing
them to every later engine as inputs, each later TensorRT engine that reads one gets its own
copy of the operations that compute it. All three partitioners do this with
``recompute_input_only_nodes`` (``torch_tensorrt/dynamo/partitioning/common.py``).

Why:

* It removes engine inputs that can be large. A 4096-token float attention mask is 64 MiB, and
  every layer reads it. On a stack of Gemma 2B-sized layers split one layer per engine (A40),
  recomputing the mask ran 6% faster and used about 1 GiB less GPU memory than passing it.
* TensorRT 11.0 to 11.4 can't build its fused attention on sm_110 (Jetson AGX Thor) for a layer
  that receives both the mask and ``cos``/``sin`` as inputs: the kernel needs 0x39174 bytes of
  shared memory, 0x38c00 at most.

What is recomputed:

* A chain starts at the nodes that read a graph input. Those cross engines as they are, so each
  graph input keeps a single reader, as GPU-resident ExecuTorch programs require.
* A chain never contains an operator that is impure (random, in place, an assert), a collective,
  an operator that runs in PyTorch, or a weight.
* A weight is identified by where it comes from, not by its size: an ``nn.Parameter``, or a
  constant that constant folding built from one (``w.T``, ``w.to(dtype)``). Any other constant
  is copied into each engine that recomputes from it, whatever its size: a buffer (rotary's
  ``inv_freq``), a tensor constant, or a value constant folding built without reading a
  parameter (the causal part of a mask, marked ``folded_without_parameters``).

The design follows the rematerialization that AOTAutograd's min-cut partitioner does between the
forward and backward graphs, recomputing values from a few saved ones instead of saving all of
them (`Min-cut optimal(*) recomputation with AOTAutograd
<https://dev-discuss.pytorch.org/t/min-cut-optimal-recomputation-i-e-activation-checkpointing-with-aotautograd/467>`_,
`min_cut_rematerialization_partition
<https://github.com/pytorch/pytorch/blob/45f9fe1c9c9997a94cd6e2afd5b93920427b566b/torch/_functorch/partitioners.py#L4178>`_).
It differs in two places:

* **Where the cut is.** AOTAutograd chooses the cut that saves the fewest bytes (`solve_min_cut
  <https://github.com/pytorch/pytorch/blob/45f9fe1c9c9997a94cd6e2afd5b93920427b566b/torch/_functorch/partitioners.py#L2491>`_).
  Here the cut is fixed at the readers of the graph inputs, which the ExecuTorch constraint
  requires anyway, and which hold the small data (padding views, position sums) that a mask or
  ``cos``/``sin`` is expanded from.
* **Which operators may be recomputed.** AOTAutograd bans recomputing random and
  compute-intensive operators and, by default, any operator outside an allowlist
  (`should_ban_recomputation
  <https://github.com/pytorch/pytorch/blob/45f9fe1c9c9997a94cd6e2afd5b93920427b566b/torch/_functorch/partitioners.py#L2585>`_,
  `get_default_op_list
  <https://github.com/pytorch/pytorch/blob/45f9fe1c9c9997a94cd6e2afd5b93920427b566b/torch/_functorch/partitioners.py#L3224>`_).
  Those lists don't describe Torch-TensorRT's lowered graphs. Lowering keeps ``aten.matmul``,
  which isn't among the compute-intensive operators (``mm``, ``bmm``, ...), and the allowlist
  leaves out ``matmul`` and ``cat``, which Hugging Face rotary embeddings use to build
  ``cos``/``sin``. Recomputing a weightless chain is taken to be cheap instead.

What it costs:

* Small values are recomputed too. With a 64-token prompt, recomputing a 16 KiB mask and 32 KiB
  of ``cos``/``sin`` in each engine ran about 3% slower than passing them (A40).
* A large constant that isn't a weight, such as an old GPT-2-style causal buffer of
  ``max_positions²`` entries, is copied into every engine that reads it.
* A value computed in PyTorch, or passed in as a graph input, still reaches the engines as an
  input.

Atomic Subgraphs and Fusion Preservation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Certain operator sequences are registered as **atomic subgraphs** — patterns that
should never be split across a partition boundary because TensorRT fuses them into a
single kernel. The built-in atomic patterns are:

* ``Conv → BatchNorm → {SiLU | GELU | ReLU | Sigmoid}``
* ``Conv → {SiLU | GELU | ReLU | Sigmoid}``
* ``Mul → Add``
* ``Mul → Mul``

Custom patterns can be registered with:

.. code-block:: python

    from torch_tensorrt.dynamo.partitioning._atomic_subgraphs import register_atomic_subgraph

    @register_atomic_subgraph(init_args=(torch.nn.ReLU(),), is_core_aten=True)
    class MyFusedPattern(torch.nn.Module):
        def forward(self, x, w):
            return torch.ops.aten.relu(torch.ops.aten.mm(x, w))

During resource partitioning, ``get_node_in_fusion_pattern()`` identifies all fusion
groups in the graph and the splitter checks every candidate split point against these
groups, migrating whole fusions to keep them together.

User-Facing Controls
^^^^^^^^^^^^^^^^^^^^^

All partitioning behaviour is controlled via ``CompilationSettings``:

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Parameter
     - Default
     - Effect on partitioning
   * - ``min_block_size``
     - ``5``
     - TRT subgraphs smaller than this are merged back into PyTorch.
   * - ``torch_executed_ops``
     - ``set()``
     - Force these ``torch.ops`` targets to PyTorch regardless of converter support.
   * - ``require_full_compilation``
     - ``False``
     - Raise if any node cannot be placed in TensorRT.
   * - ``assume_dynamic_shape_support``
     - ``False``
     - Skip per-converter dynamic shape checks; treat all converters as dynamic-capable.
   * - ``enable_resource_partitioning``
     - ``False``
     - Run resource partitioning after capability partitioning.
   * - ``cpu_memory_budget``
     - ``None``
     - Host memory budget for the build; each engine gets ``(budget - RSS) / 4``. ``None`` uses available system memory.
   * - ``max_partition_size``
     - ``None``
     - Weight bytes per engine: a finer control than ``cpu_memory_budget``, which it replaces
       when set. The split then depends only on the model and this size.

----
