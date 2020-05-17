Compiler Phases
----------------

.. toctree::
    :caption: Compiler Phases
    :maxdepth: 1
    :hidden:

    lowering
    conversion
    execution

Lowering
^^^^^^^^^^^
:ref:`lowering`

The lowering is made up of a set of passes (some from PyTorch and some specific to TRTorch)
run over the graph IR to map the large PyTorch opset to a reduced opset that is easier to convert to
TensorRT.

Conversion
^^^^^^^^^^^
:ref:`conversion`

In the conversion phase we traverse the lowered graph and construct an equivalent TensorRT graph.
The conversion phase is made up of three main components, a context to manage compile time data,
a evaluator library which will execute operations that can be resolved at compile time and a converter
library which maps an op from JIT to TensorRT.

Execution
^^^^^^^^^^^
:ref:`execution`

The execution phase constructs a TorchScript program to run the converted TensorRT engine. It
takes a serialized engine and instantiates it within a engine manager, then the compiler will
build out a JIT graph that references this engine and wraps it in a module to return to the user.
When the user executes the module, the JIT program will look up the engine and pass the inputs
to it, then return the results.