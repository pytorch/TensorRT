.. _partitioning:

Partitioning Phase
====================

The phase is optional and enabled by the user. It instructs the compiler to separate nodes into ones that should run in PyTorch and ones that should run in TensorRT.
Criteria for separation include: Lack of a converter, operator is explicitly set to run in PyTorch by the user or the node has a flag which tells partitioning to
run in PyTorch by the module fallback passes.

On a high level, Torch-TensorRT partitioning phase does the following:

* Segmentation. Go through the set of operators in order and verify if there is converter for each operator. Then, roughly separate the graph into parts that Torch-TensorRT can support and parts Torch-TensorRT cannot.

* Dependency Analysis. For every to be compiled operator there is a "complete dependency graph", which means that every input can to traced back to an input as Tensor or TensorList. Go through all segments after segmentation then do dependency analysis to ensure that there are only Tensor/TensorList inputs and outputs for TensorRT segments.

* Shape Analysis. For each segments, figure out the input and outputs shapes starting from the provided input shape from the user. Shapes can be calculated by running the graphs with JIT.

* Conversion. Every TensorRT segments will be converted to TensorRT engine. This part is done in compiler.cpp, but it's still a phase in our partitioning process.

* Stitching. Stitch all TensorRT engines with PyTorch nodes altogether.

Here are the brief description of these functions of each file:

PartitonInfo.h/.cpp
***********************************

    `core/partitioning/PartitionInfo.h <https://github.com/pytorch/TensorRT/blob/master/core/partitioning/PartitionInfo.h>`_

The automatic fallback APIs that is used for partitioning.


SegmentedBlock.h/.cpp
***********************************

    `core/partitioning/SegmentedBlock.h <https://github.com/pytorch/TensorRT/blob/master/core/partitioning/SegmentedBlock.h>`_

The main data structures that is used to maintain information for each segments after segmentation.


shape_analysis.h/.cpp
***********************************

    `core/partitioning/shape_analysis.h <https://github.com/pytorch/TensorRT/blob/master/core/partitioning/shape_analysis.h>`_

Code implementation to get the shapes for each segments by running them in JIT.


partitioning.h/.cpp
***********************************
    `core/partitioning/partitioning.h <https://github.com/pytorch/TensorRT/blob/master/core/partitioning/partitioning.h>`_

APIs and main code implementation for partitioning phase.

Automatic Fallback
====================

To enable automatic fallback feature, you can set following attributes in Python:

.. code-block:: none

      import torch
      import torch_tensorrt as torchtrt

      ...
      model = MyModel()
      ts_model = torch.jit.script(model)
      trt_model = torchtrt.ts.compile(model, **{
        ...
        "min_block_size" : 3,
        "torch_executed_ops": ["aten::add"],
        "torch_executed_modules": [],
      })

* enabled: By default automatic fallback will be off. It is enabled by setting it to True.
* min_block_size: The minimum number of consecutive operations that must satisfy to be converted to TensorRT. For example, if it's set to 3, then there must be 3 consecutive supported operators then this segments will be converted.
* forced_fallback_ops: A list of strings that will be the names of operations that the user explicitly want to be in PyTorch nodes.


.. code-block:: none

      #include "torch/script.h"
      #include "torch_tensorrt/torch_tensorrt.h"

      ...
      auto in = torch::randn({1, 3, 224, 224}, {torch::kCUDA});

      auto mod = torch::jit::load("trt_ts_module.ts");
      auto input_sizes =  std::vector<torchtrt::InputRange>{{in.sizes()}};
      torchtrt::ts::CompileSpec cfg(input_sizes);
      cfg.min_block_size = 2;
      cfg.torch_executed_ops.push_back("aten::relu");
      auto trt_mod = torchtrt::ts::compile(mod, cfg);
      auto out = trt_mod.forward({in});

Dependency Aware Partitioning
====================
During segmentation, Torch-TensorRT uses a dependency graph of the input TorchScript nodes to reduce the number of segments created. Consider this example from test Partitioning.SegmentModelWithDependencyAwareness in `tests/core/partitioning/test_segmentation.cpp <https://github.com/pytorch/TensorRT/blob/master/tests/core/partitioning/test_segmentation.cpp>`_

.. code-block:: none

    graph(%x : Tensor, %y : Tensor):
        %3 : int = prim::Constant[value=0]()
        %20 : int = prim::Constant[value=1]()
        %add : Tensor = aten::add(%x, %y, %20)
        %x_lgamma : Tensor = aten::lgamma(%x)
        %mul : Tensor = aten::mul(%x, %y)
        %y_lgamma : Tensor = aten::lgamma(%y)
        %div : Tensor = aten::div(%x, %y)
        %div_lgamma : Tensor = aten::lgamma(%div)
        %27 : Tensor[] = prim::ListConstruct(%x_lgamma, %y_lgamma, %div_lgamma, %add, %mul)
        %12 : Tensor = aten::cat(%27, %3)
        return (%12)

In this graph `aten::lgamma` is not supported by conversion and must be partitioned in a Torch fallback segment. If Torch-TensorRT uses a greedy segmentation strategy that traverses nodes in the input graph in order and gathers ops with the same target (TensorRT or Torch) into a segment until it encounters an op with a different target, the resulting partition includes 7 segments, many with just a single op.

.. code-block:: none

    Segment Block @0:
        Target: TensorRT

        Graph: graph(%x : Tensor,
            %y : Tensor):
        %3 : int = prim::Constant[value=1]()
        %0 : Tensor = aten::add(%x, %y, %3)
        return ()

    Segment Block @1:
        Target: Torch

        Graph: graph(%x : Tensor):
        %0 : Tensor = aten::lgamma(%x)
        return ()

    Segment Block @2:
        Target: TensorRT

        Graph: graph(%x : Tensor,
            %y : Tensor):
        %0 : Tensor = aten::mul(%x, %y)
        return ()

    Segment Block @3:
        Target: Torch

        Graph: graph(%y : Tensor):
        %0 : Tensor = aten::lgamma(%y)
        return ()

    Segment Block @4:
        Target: TensorRT

        Graph: graph(%x : Tensor,
            %y : Tensor):
        %0 : Tensor = aten::div(%x, %y)
        return ()

    Segment Block @5:
        Target: Torch

        Graph: graph(%1 : Tensor):
        %0 : Tensor = aten::lgamma(%1)
        return ()

    Segment Block @6:
        Target: TensorRT

        Graph: graph(%1 : Tensor,
            %2 : Tensor,
            %3 : Tensor,
            %4 : Tensor,
            %5 : Tensor):
        %7 : int = prim::Constant[value=0]()
        %0 : Tensor[] = prim::ListConstruct(%1, %2, %3, %4, %5)
        %6 : Tensor = aten::cat(%0, %7)
        return ()

This partition is valid, but the segmentation is suboptimal. These arithmetic ops and `aten::lgamma` ops are each split into their own segment as we alternate between Torch and TensorRT targets in the linear traversal of the graph.

.. code-block:: none

    %add : Tensor = aten::add(%x, %y, %20)
    %x_lgamma : Tensor = aten::lgamma(%x)
    %mul : Tensor = aten::mul(%x, %y)
    %y_lgamma : Tensor = aten::lgamma(%y)
    %div : Tensor = aten::div(%x, %y)
    %div_lgamma : Tensor = aten::lgamma(%div)

Each of the arithmetic ops in this segment is only dependent on constants and the inputs `%x` and `%y`. The `aten::lgamma` ops are dependent on the inputs `%x`, `%y` and the output of the `aten::div`. This means that we could rewrite this portion of the input graph as below without changing the behavior of the graph. This reordered series of ops could be cleanly partitioned into just 2 segments using the greedy segmentation approach described above.

.. code-block:: none

    %add : Tensor = aten::add(%x, %y, %20)
    %mul : Tensor = aten::mul(%x, %y)
    %div : Tensor = aten::div(%x, %y)
    %x_lgamma : Tensor = aten::lgamma(%x)
    %y_lgamma : Tensor = aten::lgamma(%y)
    %div_lgamma : Tensor = aten::lgamma(%div)

By adding awareness of the dependencies between ops to the basic greedy segmentation approach we can achieve the same partition without rewriting the graph. Now we will maintain both Torch and TensorRT targeted segments at the same time as we traverse the graph. We will only finalize a segment once we hit an op that is both dependent on an op in the segment and has a different target. This will allow the partition to create larger segments by reordering nodes across the segment boundary while guaranteeing that we will not modify the behavior of the graph by reordering nodes relative to their dependencies.
In this example we will collect the arithmetic ops in a TensorRT segment and the `aten::lgamma` ops in a Torch segment. When we encounter the `%div_lgamma : Tensor = aten::lgamma(%div)` op we can see it is dependent on `%div : Tensor = aten::div(%x, %y)` in the current TensorRT segment. This triggers finalization of the TensorRT segment containing the `aten::div` op to guarantee it will appear before its dependency in the final partition. The Torch segment containing the `aten::lgamma` op is finalized when we encounter the `prim::ListConstruct` op which targets TensorRT and is dependent on the results of the `aten::lgamma` ops.

.. code-block:: none

    Segment Block @0:
        Target: TensorRT

        Graph: graph(%x : Tensor,
            %y : Tensor):
        %3 : int = prim::Constant[value=1]()
        %0 : Tensor = aten::add(%x, %y, %3)
        %4 : Tensor = aten::mul(%x, %y)
        %5 : Tensor = aten::div(%x, %y)
        return ()

    Segment Block @1:
        Target: Torch

        Graph: graph(%x : Tensor,
            %y : Tensor,
            %5 : Tensor):
        %0 : Tensor = aten::lgamma(%x)
        %2 : Tensor = aten::lgamma(%y)
        %4 : Tensor = aten::lgamma(%5)
        return ()

    Segment Block @2:
        Target: TensorRT

        Graph: graph(%1 : Tensor,
            %2 : Tensor,
            %3 : Tensor,
            %4 : Tensor,
            %5 : Tensor):
        %7 : int = prim::Constant[value=0]()
        %0 : Tensor[] = prim::ListConstruct(%1, %2, %3, %4, %5)
        %6 : Tensor = aten::cat(%0, %7)
        return ()

In some cases this approach may create adjacent segments in the partition which have the same target. As a clean-up step we can consolidate these adjacent segments to further reduce the number of segments in the final partition.
The merge segments step identifies a list of segments that are adjacent in the graph, have the same target, and are not marked as `do_not_merge`. The nodes from these segments will be combined into a single new segment that will replace the merged segments in the partition.
The `do_not_merge` marking is used to prevent merging of segments created for conditional nodes and loops that are handled as special cases in graph stitching and should not be merged with adjacent segments of the same type. 
