#include <executorch/backends/cuda/runtime/cuda_allocator.h>
#include <executorch/runtime/core/device_allocator.h>

namespace {

// A program delegated to TensorRT still asks the ExecuTorch runtime for its
// device planned buffers, and the runtime can only serve those once something
// has registered a CUDA DeviceAllocator. The pinned ExecuTorch release does
// that from inside the CUDA/AOTI delegate, which an application that delegates
// to TensorRT alone has no reason to link. Register it here instead.
//
// Deliberately unguarded. If the CUDA/AOTI delegate is ever linked into the
// same binary it registers the same singleton a second time, and the registry
// is meant to abort on that rather than silently pick one.
[[maybe_unused]] const bool cuda_device_allocator_registered = [] {
  executorch::runtime::register_device_allocator(&executorch::backends::cuda::CudaAllocator::instance());
  return true;
}();

} // namespace
