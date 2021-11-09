Compiler Phases
----------------

.. toctree::
    :caption: Compiler Phases
    :maxdepth: 1
    :hidden:

    lowering
    partitioning
    conversion
    runtime

Lowering
^^^^^^^^^^^
:ref:`lowering`

The lowering is made up of a set of passes (some from PyTorch and some specific to Torch-TensorRT)
run over the graph IR to map the large PyTorch opset to a reduced opset that is easier to convert to
TensorRT.

Partitioning
^^^^^^^^^^^^^
:ref:`partitioning

The phase is optional and enabled by the user. It instructs the compiler to seperate nodes into ones that should run in PyTorch and ones that should run in TensorRT.
Criteria for seperation include: Lack of a converter, operator is explicitly set to run in PyTorch by the user or the node has a flag which tells partitioning to
run in PyTorch by the module fallback passes.

Conversion
^^^^^^^^^^^
:ref:`conversion`

In the conversion phase we traverse the lowered graph and construct an equivalent TensorRT graph.
The conversion phase is made up of three main components, a context to manage compile time data,
a evaluator library which will execute operations that can be resolved at compile time and a converter
library which maps an op from JIT to TensorRT.

Compilation and Runtime
^^^^^^^^^^^^^^^^^^^^^^^^
:ref:`runtime`

The final compilation phase constructs a TorchScript program to run the converted TensorRT engine. It
takes a serialized engine and instantiates it within a engine manager, then the compiler will
build out a JIT graph that references this engine and wraps it in a module to return to the user.
When the user executes the module, the JIT program run in the JIT runtime extended by Torch-TensorRT with the data providied from the user.
