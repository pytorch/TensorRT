#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>

namespace torch {
namespace executor {
namespace native {

using executorch::aten::Tensor;
using executorch::runtime::KernelRuntimeContext;

// ExecuTorch publishes no header for its portable kernel sources, and the
// Bazel target that owns this file compiles op__device_copy.cpp straight from
// the pinned tree, so the two entry points are declared here.
Tensor& _h2d_copy_out(KernelRuntimeContext& ctx, const Tensor& self, Tensor& out);
Tensor& _d2h_copy_out(KernelRuntimeContext& ctx, const Tensor& self, Tensor& out);

} // namespace native
} // namespace executor
} // namespace torch

// A program delegated to TensorRT still runs two ops outside the delegate: the
// host to device copy of its inputs and the device to host copy of its outputs,
// which the device placement pass inserts at export time. ExecuTorch registers
// their kernels from a generated kernel library, which this Bazel build does
// not produce, so a runner linking the core runtime alone fails every
// load_method with
//
//   kernel 'et_copy::_h2d_copy.out' not found.
//
// Register those two rather than the whole portable kernel set, which a fully
// delegated program never calls.
EXECUTORCH_LIBRARY(et_copy, "_h2d_copy.out", torch::executor::native::_h2d_copy_out);
EXECUTORCH_LIBRARY(et_copy, "_d2h_copy.out", torch::executor::native::_d2h_copy_out);
