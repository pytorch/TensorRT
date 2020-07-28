.. _execution:

Execution Phase
================

The execution phase is responsible for constructing self standing TorchScript graphs with embedded TensorRT engines and serving as the runtime
when these engines are called. The main interface accepts a serialized TensorRT engine. The execution phase
will deserialize and wrap this engine in a class which maintains a execution context for each engine
and some metadata about its inputs and outputs and is compatable with the TorchScript interpreter so that
it can be moved around and used like other TorchScript IValues. The engine is run by providing it and inputs
to the ``trt::execute_engine`` operator which will take the engine and its inputs and return the results of engine exeuction.


Background
------------
PyTorch JIT's runtime is based around a stack machine, all operators pop off arguments from the stack, pass them to
some implementation of the operator then push results back onto the stack. The actual elements of the stack
are ``torch::jit::IValues``, the same type we evaluate in the conversion phase (the realization of the abstract
torch::jit::Value type).

TensorRT Engine Executor Op
----------------------------

When the TRTorch is loaded, it registers an operator in the PyTorch JIT operator library called
``trt::execute_engine(Tensor[] inputs, __torch__.torch.classes.tensorrt.Engine engine) -> Tensor[]`` which takes an
instantiated engine and list of inputs. Compiled graphs store this engine in an attribute so that it is portable and serializable.
When the op is called, an instnantiated engine and input tensors are popped off the runtime stack. These inputs are passed into a generic engine execution function which
will run the tensors through the TensorRT engine and return new tensors as results. These tensors are pushed on to the
stack so that the next op whatever it is can use it.

Constructing the Resulting Graph
-----------------------------------

Once the engine is deserialized and instantiated, the compiler will construct a graph that will execute the engine when the module is called.
Here is an example:

.. code-block::

    graph(%self_1 : __torch__.torchvision.models.resnet.___torch_mangle_4847.ResNet_trt,
      %input_0 : Tensor):
        %1 : __torch__.torch.classes.tensorrt.Engine = prim::GetAttr[name="__torch___torchvision_models_resnet____torch_mangle_4847_ResNet_trt_engine"](%self_1)
        %3 : Tensor[] = prim::ListConstruct(%input_0)
        %4 : Tensor[] = trt::execute_engine(%3, %1)
        %5 : Tensor = prim::ListUnpack(%4)
    return (%5)

You can see the engine attribute in the graph and the ``trt::execute_engine`` op taking a list of input tensors and an engine in
and produces a list of output tensors which is returned. When ``forward`` is called on the module this graph is executed, thereby
running the TensorRT engine.

In the case of multiple outputs, the compiled graph may repack output tensors into a Tuple to return back to the user.

.. code-block::

    graph(%self_1 : __torch__.PyTorch.Detection.SSD.src.model.SSD300_trt,
      %input_0 : Tensor):
        %1 : __torch__.torch.classes.tensorrt.Engine = prim::GetAttr[name="__torch___PyTorch_Detection_SSD_src_model_SSD300_trt_engine"](%self_1)
        %3 : Tensor[] = prim::ListConstruct(%input_0)
        %4 : Tensor[] = trt::execute_engine(%3, %1)
        %5 : Tensor, %6 : Tensor = prim::ListUnpack(%4)
        %7 : (Tensor, Tensor) = prim::TupleConstruct(%5, %6)
    return (%7)

Serialization and Deserialization
----------------------------------

Serialization and deserialization of TensorRT engines embedded in TorchScript graphs are handled by the holder class for the engine and TorchBind.
When a TorchScript module is saved, the pickler will run serilization on the cuda engine and store the serialized engine in the zip file created.
When deserializing, the depickler will call a constructor for the engine holder class with the serialized engine so that it can be set up again for
execution.