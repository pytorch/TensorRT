# Evaluators

   Operators whose outputs are known at compile time are considered to be evaluators. For example, in PyTorch library,
   operations like `aten::zeros`, `aten::full` are constants in the graph which can be expressed as constant values in TensorRT
   graph. Pytorch library expresses constants and new data structures using `aten` and `prim` libraries. Corresponding TensorRT
   implementations for such operators can be found in `aten.cpp` and `prim.cpp`.
