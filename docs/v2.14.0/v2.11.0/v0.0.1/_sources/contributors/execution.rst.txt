.. _execution:

Execution Phase
================

The execution phase is responsible for managing TensorRT engines, constructing a new module for the TensorRT engines,
and acting as a runtime for JIT modules calling TensorRT engines. The main interface accepts a serialized
TensorRT engine. It stands up the engine within the Engine Manager which maintains a execution context for each engine
and some metadata about its inputs and outputs. Each engine is assigned an ID which can be used to reference the engine
when running the a module with the JIT interpreter.

Background
------------
PyTorch JIT's runtime is based around a stack machine, all operators pop off arguments from the stack, pass them to
some implementation of the operator then push results back onto the stack. The actual elements of the stack
are ``torch::jit::IValues``, the same type we evaluate in the conversion phase (the realization of the abstract
torch::jit::Value type).

TensorRT Engine Executor Op
----------------------------

When the TRTorch is loaded, it registers an operator in the PyTorch JIT operator library called ``trt::execute_engine(int id, ...) -> ...``
which takes a engine ID and inputs. It will then use the ID to look up the coresponding execution context, then
pop off the inputs from the runtime stack. These inputs are passed into a generic engine execution function which
will run the tensors through the TensorRT engine and return new tensors as results. These tensors are pushed on to the
stack so that the next op whatever it is can use it.

Constructing the Resulting Graph
-----------------------------------

Once the engine is registered, the compiler will construct a graph that will execute the engine when the module is called.
Here is an example:

.. code-block::

    graph(%self.1 : __torch__.___torch_mangle_10.LeNet_trt,
        %2 : Tensor):
        %1 : int = prim::Constant[value=94106001690080]()
        %3 : Tensor = trt::execute_engine(%1, %2)
        return (%3)
    (AddEngineToGraph)

You can see the ID as a constant in the graph and the ``trt::execute_engine`` op taking the constant and an input tensor in
and produces an output tensor which is returned. When ``forward`` is called on the module this graph is executed, thereby
running the TensorRT engine.