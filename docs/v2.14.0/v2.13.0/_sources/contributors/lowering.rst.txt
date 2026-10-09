.. _lowering:

Lowering Phase
===============

The lowering phase maps the captured FX/TorchScript graph from its original opset down
to a form that the conversion stage can translate 1-to-1 into TensorRT operations. The
goal is to reduce the converter library to a small set of simple, well-defined
translations rather than requiring each converter to handle complex multi-op patterns.

Dynamo Lowering (Primary Path)
--------------------------------

The Dynamo lowering phase operates on ``torch.fx.Graph`` objects and has two mechanisms:

Decompositions
^^^^^^^^^^^^^^^

The primary mechanism, covering roughly 80% of lowering work. A higher-level ATen
operator is replaced inline by an equivalent subgraph of
`Core ATen operators <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_ir.html>`_
using a plain PyTorch function. This is easy to write because the body uses normal
PyTorch code and is automatically shape-propagating via
`FakeTensor <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_fake_tensor.html>`_.

.. code-block:: python

    @register_torch_trt_decomposition(aten.linear, registry=TORCH_TRT_DECOMPOSITIONS)
    def linear_decomposition(
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        return torch.add(torch.ops.aten.mm(x, w.T), b)

Decompositions are registered per-op and are applied automatically during the lowering
pass over the graph. You can also adjust which PyTorch-builtin decompositions are
enabled or disabled via ``torch_tensorrt.dynamo.lowering.torch_enabled_decompositions``
and ``torch_tensorrt.dynamo.lowering.torch_disabled_decompositions``.

Subgraph Matching and Replacement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Used for structural rewrites that cannot be expressed as a simple per-op substitution.
Examples include:

* Inserting KV-cache operators for attention optimizations in LLM models
* Decomposing complex-number arithmetic into real/imaginary components (since TensorRT
  has no complex dtype support)
* Removing vestigial subgraphs (e.g. training-only operations that survive export)

These passes use
`torch.fx.subgraph_rewriter <https://docs.pytorch.org/docs/stable/fx.html#graph-manipulation>`_
or manual graph surgery and are registered into the pass manager alongside decompositions.

For a guide on writing new Dynamo lowering passes, see :ref:`writing_dynamo_aten_lowering_passes`.

.. note::

   After each lowering pass the graph must remain valid and callable. Use
   ``gm.graph.lint()`` and ``gm.recompile()`` after any structural changes.

----

TorchScript Lowering Passes (Legacy ``ts`` Path)
-------------------------------------------------

The following passes operate on TorchScript IR (``torch.jit.Graph``) and are used by
the legacy TorchScript frontend. They are not part of the Dynamo pipeline.

You can see the effect of each pass by setting the log level to ``Level::kGraph``.

Passes Used
-------------

EliminateCommonSubexpression
***********************************

    `torch/csrc/jit/passes/common_subexpression_elimination.h <https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/passes/common_subexpression_elimination.h>`_

Removes common subexpressions in the graph



Eliminate Dead Code
**************************

    `torch/csrc/jit/passes/dead_code_elimination.h <https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/passes/dead_code_elimination.h>`_

Dead code elimination will check if a node has side effects and not delete it if it does.

Eliminate Exception Or Pass Pattern
***************************************

    `Torch-TensorRT/core/lowering/passes/exception_elimination.cpp <https://github.com/pytorch/TensorRT/blob/master/core/lowering/passes/exception_elimination.cpp>`_

A common pattern in scripted modules are dimension guards which will throw exceptions if
the input dimension is not what was expected.

.. code-block:: none

    %1013 : bool = aten::ne(%1012, %24) # ~/.local/lib/python3.6/site-packages/torch/nn/modules/batchnorm.py:248:11
        = prim::If(%1013) # ~/.local/lib/python3.6/site-packages/torch/nn/modules/batchnorm.py:248:8
        block0():
            = prim::RaiseException(%23) # ~/.local/lib/python3.6/site-packages/torch/nn/modules/batchnorm.py:249:12
        -> ()
        block1():
        -> ()

Since we are resolving all of this at compile time and there are no exceptions in the TensorRT graph, we just remove it.

Eliminate Redundant Guards
***************************************

    `torch/csrc/jit/passes/guard_elimination.h <https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/passes/guard_elimination.h>`_

Eliminate redundant guards for ops whose outputs are fully determined by their inputs i.e. if inputs to such ops are
guarded we are allowed to remove a guard on ops' outputs

Freeze Module
***************************************

    `torch/csrc/jit/passes/freeze_module.h <https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/passes/freeze_module.h>`_

Freeze attributes and inline constants and modules. Propagates constants in the graph.

Fuse AddMM Branches
***************************************

    `Torch-TensorRT/core/lowering/passes/fuse_addmm_branches.cpp <https://github.com/pytorch/TensorRT/blob/master/core/lowering/passes/fuse_addmm_branches.cpp>`_

A common pattern in scripted modules is tensors of different dimensions use different constructions for implementing linear layers. We fuse these
different variants into a single one that will get caught by the Unpack AddMM pass.

.. code-block:: none

    %ret : Tensor = prim::If(%622)
    block0():
      %ret.1 : Tensor = aten::addmm(%self.fc.bias, %x9.1, %3677, %3, %3)
      -> (%ret.1)
    block1():
      %output.1 : Tensor = aten::matmul(%x9.1, %3677)
      %output0.1 : Tensor = aten::add_(%output.1, %self.fc.bias, %3)
      -> (%output0.1)

We fuse this set of blocks into a graph like this:

.. code-block:: none

    %ret : Tensor = aten::addmm(%self.fc.bias, %x9.1, %3677, %3, %3)

Fuse Linear
***************************************

    `torch/csrc/jit/passes/fuse_linear.h <https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/passes/fuse_linear.h>`_

Match the ``aten::linear`` pattern and fuse it into a single ``aten::linear``
This pass fuse the addmm or matmul + add generated by JIT back to linear

Fuse Flatten Linear
***************************************

    `Torch-TensorRT/core/lowering/passes/fuse_flatten_linear.cpp <https://github.com/pytorch/TensorRT/blob/master/core/lowering/passes/fuse_flatten_linear.cpp>`_

TensorRT implicitly flattens input layers into fully connected layers when they are higher than 1D. So when there is a
``aten::flatten`` -> ``aten::linear`` pattern we remove the ``aten::flatten``.

Lower Graph
***************************************

    `torch/csrc/jit/passes/lower_graph.h <https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/passes/lower_graph.h>`_

Given a graph with of a method which first argument is %self, lower it to a graph where
all attributes accesses are replaced with explicit inputs of the graph
(rather than results of prim::GetAttr executed on %self). Returns a tuple
(graph, parameters) where the last module.parameters.size() inputs to the
graph are the trainable parameters used in this method. The remaining inputs
are the true inputs to the function.

Lower Tuples
***************************************

    `torch/csrc/jit/passes/lower_tuples.h <https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/passes/lower_tuples.h>`_

* ``LowerSimpleTuples``:

Removes tuples where TupleConstruct and TupleUnpack are matched but leaves tuples in place across if statements, loops, and as inputs/outputs

* ``LowerAllTuples``:

Removes _all_ tuples and raises an error if some cannot be removed, this is used by ONNX to ensure there are not tuples before conversion, but will not work on graphs whose inputs contain tuples.

Module Fallback
*****************

    `Torch-TensorRT/core/lowering/passes/module_fallback.cpp <https://github.com/pytorch/TensorRT/blob/master/core/lowering/passes/module_fallback.cpp>`_

Module fallback consists of two lowering passes that must be run as a pair. The first pass is run before freezing to place delimiters in the graph around modules
that should run in PyTorch. The second pass marks nodes between these delimiters after freezing to signify they should run in PyTorch.

* ``NotateModuleForFallback``

Places delimiting nodes around module calls pre freezing to signify where in the graph nodes should run in PyTorch

* ``MarkNodesForFallback``

Looks for delimiters then marks all nodes between the delimiters to tell partitioning to run them in PyTorch

Peephole Optimize
***************************************

    `torch/csrc/jit/passes/peephole_optimze.h <https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/passes/ppeephole_optimze.h>`_

The intent for this optimization pass is to catch all of the small, easy to catch peephole optimizations you might be interested in doing.

Right now, it does:
    - Eliminate no-op 'expand' nodes
    - Simply x.t().t() to x


Remove Contiguous
***************************************

    `Torch-TensorRT/core/lowering/passes/remove_contiguous.cpp <https://github.com/pytorch/TensorRT/blob/master/core/lowering/passes/remove_contiguous.cpp>`_

Removes contiguous operators since we are doing TensorRT memory is already contiguous.


Remove Dropout
***************************************

    `Torch-TensorRT/core/lowering/passes/remove_dropout.cpp <https://github.com/pytorch/TensorRT/blob/master/core/lowering/passes/remove_dropout.cpp>`_

Removes dropout operators since we are doing inference.

Remove To
***************************************

    `Torch-TensorRT/core/lowering/passes/remove_to.cpp <https://github.com/pytorch/TensorRT/blob/master/core/lowering/passes/remove_to.cpp>`_

Removes ``aten::to`` operators that do casting, since TensorRT manages it itself. It is important that this is one of the last passes run so that
other passes have a change to move required cast operators out of the main namespace.

Unpack AddMM
***************************************

    `Torch-TensorRT/core/lowering/passes/unpack_addmm.cpp <https://github.com/pytorch/TensorRT/blob/master/core/lowering/passes/unpack_addmm.cpp>`_

Unpacks ``aten::addmm`` into ``aten::matmul`` and ``aten::add_`` (with an additional ``trt::const``
op to freeze the bias in the TensorRT graph). This lets us reuse the ``aten::matmul`` and ``aten::add_``
converters instead of needing a dedicated converter.

Unpack LogSoftmax
***************************************

    `Torch-TensorRT/core/lowering/passes/unpack_log_softmax.cpp <https://github.com/pytorch/TensorRT/blob/master/core/lowering/passes/unpack_log_softmax.cpp>`_

Unpacks ``aten::logsoftmax`` into ``aten::softmax`` and ``aten::log``. This lets us reuse the
``aten::softmax`` and ``aten::log`` converters instead of needing a dedicated converter.

Unroll Loops
***************************************

    `torch/csrc/jit/passes/loop_unrolling.h <https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/passes/loop_unrolling.h>`_

Unrolls the operations of compatible loops (e.g. sufficiently short) so that you only have to go through the loop once.

Replace Tile with Repeat
***************************************

    `Torch-TensorRT/core/lowering/passes/tile_to_repeat.cpp <https://github.com/pytorch/TensorRT/blob/master/core/lowering/passes/tile_to_repeat.cpp>`_

Removes dropout operators since we are doing inference.
