#include "torch_tensorrt/ptq.h"
#include "torch/torch.h"

namespace torch_tensorrt {
namespace ptq {

bool get_batch_impl(void* bindings[], const char* names[], int nbBindings, torch::Tensor& data) {
  for (int i = 0; i < nbBindings; i++) {
    data = data.to(at::kCUDA).contiguous();
    bindings[i] = data.data_ptr();
  }
  return true;
}

} // namespace ptq
} // namespace torch_tensorrt
