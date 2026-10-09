.. _dryrun:

Dryrun Mode
===========

Dryrun mode runs the full Torch-TensorRT partitioning pipeline — lowering, capability
checking, graph splitting — but stops **before** building any TRT engines. It prints a
detailed report describing exactly how the graph will be partitioned and which operators
will run in TRT vs. PyTorch.

Use dryrun to:

* Understand TRT operator coverage for a new model without waiting for compilation.
* Tune ``min_block_size`` before committing to a full compile.
* Debug why an op is falling back to PyTorch.
* Compare the partitioning effect of different ``CompilationSettings``.

----

Enabling Dryrun
---------------

Set ``dryrun=True`` to print the report to stdout:

.. code-block:: python

    import torch
    import torch_tensorrt

    exp_program = torch.export.export(model, tuple(inputs))
    torch_tensorrt.dynamo.compile(
        exp_program,
        arg_inputs=inputs,
        dryrun=True,
    )

Set ``dryrun`` to a file path to also save the report:

.. code-block:: python

    torch_tensorrt.dynamo.compile(
        exp_program,
        arg_inputs=inputs,
        dryrun="/tmp/partition_report.txt",
    )

``dryrun`` is a :class:`~torch_tensorrt.dynamo.CompilationSettings` parameter and can
also be passed via ``torch.compile``:

.. code-block:: python

    trt_model = torch.compile(model, backend="tensorrt", options={"dryrun": True})
    trt_model(*inputs)  # report printed on first forward pass

----

Reading the Report
------------------

A typical dryrun report looks like this::

    ++++++++++++++++++++++++++ Dry-Run Results for Graph ++++++++++++++++++++++++++

    The graph consists of 142 Total Operators, of which 138 operators are supported, 97.18% coverage

    The following ops are currently unsupported or excluded from conversion, and are listed with their op-count in the graph:
     torch.ops.aten.embedding.default: 1
     torch.ops.aten.index.Tensor: 3

    Compiled with: CompilationSettings(min_block_size=5, ...)

      Graph Structure:

       Inputs: List[Tensor: (1, 512)@int64]
        ...
          TRT Engine #1 - Submodule name: _run_on_acc_0
           Engine Inputs: List[Tensor: (1, 512, 768)@float32]
           Number of Operators in Engine: 135
           Engine Outputs: Tensor: (1, 512, 30522)@float32
        ...
       Outputs: List[Tensor: (1, 512, 30522)@float32]

      ------------------------- Aggregate Stats -------------------------

       Average Number of Operators per TRT Engine: 135.0
       Most Operators in a TRT Engine: 135

       ********** Recommendations **********

       - For minimal graph segmentation, select min_block_size=135 which would generate 1 TRT engine(s)
       - The current level of graph segmentation is equivalent to selecting min_block_size=5 which generates 1 TRT engine(s)

**Sections explained:**

Coverage summary
    Total operators, TRT-supported operators, and coverage percentage. "Supported" here
    means the operator has a converter registered **and** its capability validator
    passes for this specific node.

Unsupported ops
    Operators that will fall back to PyTorch, with their occurrence count. Check these
    against the converter registry or your ``torch_executed_ops`` setting.

Nodes set to run in Torch
    Specific nodes excluded from TRT blocks. A node may appear here even if it has a
    converter, if it was not included in any TRT block by the partitioner (e.g., it was
    below ``min_block_size``).

Graph structure
    ASCII schematic of input tensors → TRT engine blocks → output tensors. Each TRT
    engine block shows its input/output shapes and operator count. Use this to see
    where PyTorch↔TRT transitions occur.

Aggregate stats
    Min, max, and average operator counts per engine. More engines with fewer operators
    each means more context-switch overhead.

Recommendations
    Suggested ``min_block_size`` values and the resulting engine counts:

    * **Minimal segmentation** — the largest block absorbs the most operators; generates
      the fewest engines.
    * **Current setting** — what your current ``min_block_size`` produces.

    For models where TRT coverage is close to 100%, a single large engine is usually
    optimal. For mixed models, the recommendation helps you balance coverage vs. overhead.

----

Workflow: Tuning min_block_size
---------------------------------

.. code-block:: python

    # Step 1: run dryrun with a loose min_block_size to see the full partition map
    torch_tensorrt.dynamo.compile(
        exp_program, arg_inputs=inputs,
        dryrun="/tmp/report_loose.txt",
        min_block_size=1,
    )

    # Step 2: read recommendations in the report, pick an appropriate value
    # Step 3: compile for real
    trt_gm = torch_tensorrt.dynamo.compile(
        exp_program, arg_inputs=inputs,
        min_block_size=10,
    )

----

Debugging Fallback Ops
-----------------------

If an op you expect to be TRT-supported appears in the unsupported list:

1. Check that a converter is registered:

   .. code-block:: python

       from torch_tensorrt.dynamo.conversion._ConverterRegistry import DYNAMO_CONVERTERS
       print(torch.ops.aten.embedding.default in DYNAMO_CONVERTERS)

2. Check the converter's capability validator:

   .. code-block:: python

       from torch_tensorrt.dynamo.partitioning import get_graph_converter_support
       n_supported, n_total = get_graph_converter_support(gm, torch_executed_ops=set())
       print(f"{n_supported}/{n_total} ops supported")

3. Check ``torch_executed_ops`` — the op may be explicitly forced to PyTorch.

4. Check ``min_block_size`` — the block containing the op may have been merged back into
   PyTorch because it had too few operators. Reduce ``min_block_size`` in dryrun to
   confirm.

----

Saving the Report
-----------------

Pass a string path to ``dryrun`` to persist the report:

.. code-block:: python

    torch_tensorrt.dynamo.compile(
        exp_program, arg_inputs=inputs, dryrun="/tmp/report.txt"
    )

If the file already exists, a warning is logged and the file is **not** overwritten.
Remove the old file manually before rerunning.

The dryrun output is also available at ``DEBUG`` log level even when ``dryrun=False``:

.. code-block:: python

    import logging
    logging.getLogger("torch_tensorrt").setLevel(logging.DEBUG)
