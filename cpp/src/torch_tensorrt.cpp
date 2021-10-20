#include "torch/csrc/jit/api/module.h"

#include "core/compiler.h"
#include "core/util/prelude.h"

#include "torch_tensorrt/torch_tensorrt.h"

namespace torch_tensorrt {
// Defined in types.cpp
trtorch::core::runtime::CudaDevice to_internal_cuda_device(Device device);
namespace torchscript {
// Defined in compile_spec.cpp
trtorch::core::CompileSpec to_internal_compile_spec(CompileSpec external);

bool CheckMethodOperatorSupport(const torch::jit::script::Module& module, std::string method_name) {
  return trtorch::core::CheckMethodOperatorSupport(module, method_name);
}

std::string ConvertGraphToTRTEngine(
    const torch::jit::script::Module& module,
    std::string method_name,
    CompileSpec info) {
  LOG_DEBUG(get_build_info());
  // Want to export a much simpler (non TRT header dependent) API so doing the
  // type conversion here
  return trtorch::core::ConvertGraphToTRTEngine(module, method_name, to_internal_compile_spec(info));
}

torch::jit::script::Module CompileGraph(const torch::jit::script::Module& module, CompileSpec info) {
  LOG_DEBUG(get_build_info());
  // Want to export a much simpler (non TRT header dependent) API so doing the
  // type conversion here
  return trtorch::core::CompileGraph(module, to_internal_compile_spec(info));
}

torch::jit::Module EmbedEngineInNewModule(const std::string& engine, Device device) {
  return trtorch::core::EmbedEngineInNewModule(engine, to_internal_cuda_device(device));
}

} //namespace ts

std::string get_build_info() {
  auto info = trtorch::core::util::get_build_info();
  return std::string("TRTorch Version: ") + TORCH_TENSORRT_VERSION + '\n' + info;
}

void dump_build_info() {
  std::cout << get_build_info() << std::endl;
}

void set_device(const int gpu_id) {
  // Want to export a much simpler (non CUDA header dependent) API
  trtorch::core::set_device(gpu_id);
}
} // namespace trtorch
