#include "torch/csrc/jit/api/module.h"

#include "core/compiler.h"
#include "core/util/prelude.h"

#include "torch_tensorrt/torch_tensorrt.h"

namespace torch_tensorrt {
// Defined in types.cpp
torch_tensorrt::core::runtime::RTDevice to_internal_rt_device(Device device);
namespace torchscript {
// Defined in compile_spec.cpp
torch_tensorrt::core::CompileSpec to_internal_compile_spec(CompileSpec external, bool converting_to_trt_engine = false);

bool check_method_operator_support(const torch::jit::script::Module& module, std::string method_name) {
  return torch_tensorrt::core::CheckMethodOperatorSupport(module, method_name);
}

std::string convert_method_to_trt_engine(
    const torch::jit::script::Module& module,
    std::string method_name,
    CompileSpec info) {
  LOG_DEBUG(get_build_info());
  // Want to export a much simpler (non TRT header dependent) API so doing the
  // type conversion here
  return torch_tensorrt::core::ConvertGraphToTRTEngine(
      module, method_name, to_internal_compile_spec(info, /*bool converting_to_trt_engine=*/true));
}

torch::jit::script::Module compile(const torch::jit::script::Module& module, CompileSpec info) {
  LOG_DEBUG(get_build_info());
  // Want to export a much simpler (non TRT header dependent) API so doing the
  // type conversion here
  return torch_tensorrt::core::CompileGraph(module, to_internal_compile_spec(info));
}

torch::jit::Module embed_engine_in_new_module(
    const std::string& engine,
    Device device,
    const std::vector<std::string>& input_binding_names,
    const std::vector<std::string>& output_binding_names) {
  return torch_tensorrt::core::EmbedEngineInNewModule(
      engine, to_internal_rt_device(device), input_binding_names, output_binding_names);
}

} // namespace torchscript

std::string get_build_info() {
  auto info = torch_tensorrt::core::util::get_build_info();
  return std::string("Torch-TensorRT Version: ") + TORCH_TENSORRT_VERSION + '\n' + info;
}

void dump_build_info() {
  std::cout << get_build_info() << std::endl;
}

void set_device(const int gpu_id) {
  // Want to export a much simpler (non CUDA header dependent) API
  torch_tensorrt::core::set_device(gpu_id);
}

static auto tensorrt_input_container = torch::class_<Input>("_torch_tensorrt", "Input").def(torch::init<>());
} // namespace torch_tensorrt
