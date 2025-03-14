#include <codecvt>

#include "core/runtime/Platform.h"
#include "core/runtime/runtime.h"
#include "core/util/macros.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

std::string serialize_bindings(const std::vector<std::string>& bindings) {
  std::stringstream ss;
  for (size_t i = 0; i < bindings.size() - 1; i++) {
    ss << bindings[i] << TRTEngine::BINDING_DELIM;
  }
  ss << bindings[bindings.size() - 1];

  std::string serialized_binding_info = ss.str();

  LOG_DEBUG("Serialized Binding Info: " << serialized_binding_info);

  return serialized_binding_info;
}

static const std::string sym_table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"; //=
std::string base64_encode(const std::string& in) {
  std::string out;
  int64_t val = 0, valb = -6;
  for (unsigned char c : in) {
    val = (val << 8) + c;
    valb += 8;
    while (valb >= 0) {
      out.push_back(sym_table[(val >> valb) & 0x3F]);
      valb -= 6;
    }
  }
  if (valb > -6) {
    out.push_back(sym_table[((val << 8) >> (valb + 8)) & 0x3F]);
  };
  while (out.size() % 4) {
    out.push_back('=');
  }
  return out;
}

std::string base64_decode(const std::string& in) {
  std::string out;
  std::vector<int> T(256, -1);
  for (int i = 0; i < 64; i++) {
    T[sym_table[i]] = i;
  }

  int64_t val = 0, valb = -8;
  for (unsigned char c : in) {
    if (T[c] == -1) {
      break;
    }
    val = (val << 6) + T[c];
    valb += 6;
    if (valb >= 0) {
      out.push_back(char((val >> valb) & 0xFF));
      valb -= 8;
    }
  }
  return out;
}

namespace {
// TODO: Implement a call method
// c10::List<at::Tensor> TRTEngine::Run(c10::List<at::Tensor> inputs) {
//     auto input_vec = inputs.vec();
//    auto output_vec = RunCudaEngine(exec_ctx, num_io, input_vec);
//
//     return c10::List<at::Tensor>(output_vec);
// }
static auto TORCHTRT_UNUSED TRTEngineTSRegistrtion =
    torch::class_<TRTEngine>("tensorrt", "Engine")
        .def(torch::init<std::vector<std::string>>())
        // TODO: .def("__call__", &TRTEngine::Run)
        // TODO: .def("run", &TRTEngine::Run)
        .def("__str__", &TRTEngine::to_str)
        .def("__repr__", &TRTEngine::to_str)
        .def("__obj_flatten__", &TRTEngine::__obj_flatten__)
        .def("enable_profiling", &TRTEngine::enable_profiling)
        .def("disable_profiling", &TRTEngine::disable_profiling)
        .def_readwrite("profile_path_prefix", &TRTEngine::profile_path_prefix)
        .def("dump_engine_layer_info_to_file", &TRTEngine::dump_engine_layer_info_to_file)
        .def("dump_engine_layer_info", &TRTEngine::dump_engine_layer_info)
        .def("get_engine_layer_info", &TRTEngine::get_engine_layer_info)
        .def("infer_outputs", &TRTEngine::infer_outputs)
        .def_readwrite("use_pre_allocated_outputs", &TRTEngine::use_pre_allocated_outputs)
        .def_readwrite("use_output_allocator_outputs", &TRTEngine::use_output_allocator_outputs)
        .def_property(
            "device_memory_budget",
            &TRTEngine::get_device_memory_budget,
            &TRTEngine::set_device_memory_budget)
        .def_property("streamable_device_memory_budget", &TRTEngine::get_streamable_device_memory_budget)
        .def_property("automatic_device_memory_budget", &TRTEngine::get_automatic_device_memory_budget)
        .def_pickle(
            [](const c10::intrusive_ptr<TRTEngine>& self) -> std::vector<std::string> { return self->serialize(); },
            [](std::vector<std::string> serialized_info) -> c10::intrusive_ptr<TRTEngine> {
              serialized_info[ENGINE_IDX] = base64_decode(serialized_info[ENGINE_IDX]);
              TRTEngine::verify_serialization_fmt(serialized_info);
              return c10::make_intrusive<TRTEngine>(serialized_info);
            });

TORCH_LIBRARY(tensorrt, m) {
  m.def("execute_engine(Tensor[] input_tensors, __torch__.torch.classes.tensorrt.Engine engine) -> Tensor[]");
  m.def("SERIALIZED_ENGINE_BINDING_DELIM", []() -> std::string { return std::string(1, TRTEngine::BINDING_DELIM); });
  m.def("SERIALIZED_RT_DEVICE_DELIM", []() -> std::string { return DEVICE_INFO_DELIM; });
  m.def("ABI_VERSION", []() -> std::string { return ABI_VERSION; });
  m.def("get_multi_device_safe_mode", []() -> bool { return MULTI_DEVICE_SAFE_MODE; });
  m.def("set_multi_device_safe_mode", [](bool multi_device_safe_mode) -> void {
    MULTI_DEVICE_SAFE_MODE = multi_device_safe_mode;
  });
  m.def("get_cudagraphs_mode", []() -> int64_t { return CUDAGRAPHS_MODE; });
  m.def("set_cudagraphs_mode", [](int64_t cudagraphs_mode) -> void {
    CUDAGRAPHS_MODE = CudaGraphsMode(cudagraphs_mode);
  });
  m.def("set_logging_level", [](int64_t level) -> void {
    util::logging::get_logger().set_reportable_log_level(util::logging::LogLevel(level));
  });
  m.def(
      "get_logging_level", []() -> int64_t { return int64_t(util::logging::get_logger().get_reportable_log_level()); });
  m.def("ABI_TARGET_IDX", []() -> int64_t { return ABI_TARGET_IDX; });
  m.def("NAME_IDX", []() -> int64_t { return NAME_IDX; });
  m.def("DEVICE_IDX", []() -> int64_t { return DEVICE_IDX; });
  m.def("ENGINE_IDX", []() -> int64_t { return ENGINE_IDX; });
  m.def("INPUT_BINDING_NAMES_IDX", []() -> int64_t { return INPUT_BINDING_NAMES_IDX; });
  m.def("OUTPUT_BINDING_NAMES_IDX", []() -> int64_t { return OUTPUT_BINDING_NAMES_IDX; });
  m.def("HW_COMPATIBLE_IDX", []() -> int64_t { return HW_COMPATIBLE_IDX; });
  m.def("SERIALIZED_METADATA_IDX", []() -> int64_t { return SERIALIZED_METADATA_IDX; });
  m.def("TARGET_PLATFORM_IDX", []() -> int64_t { return TARGET_PLATFORM_IDX; });
  m.def("REQUIRES_OUTPUT_ALLOCATOR_IDX", []() -> int64_t { return REQUIRES_OUTPUT_ALLOCATOR_IDX; });
  m.def("SERIALIZATION_LEN", []() -> int64_t { return SERIALIZATION_LEN; });
  m.def("_platform_linux_x86_64", []() -> std::string {
    auto it = get_platform_name_map().find(Platform::PlatformEnum::kLINUX_X86_64);
    return it->second;
  });
  m.def("_platform_linux_aarch64", []() -> std::string {
    auto it = get_platform_name_map().find(Platform::PlatformEnum::kLINUX_AARCH64);
    return it->second;
  });
  m.def("_platform_win_x86_64", []() -> std::string {
    auto it = get_platform_name_map().find(Platform::PlatformEnum::kWIN_X86_64);
    return it->second;
  });
  m.def("_platform_unknown", []() -> std::string {
    auto it = get_platform_name_map().find(Platform::PlatformEnum::kUNKNOWN);
    return it->second;
  });
  m.def("get_current_platform", []() -> std::string {
    auto it = get_platform_name_map().find(get_current_platform()._platform);
    return it->second;
  });
}

TORCH_LIBRARY_IMPL(tensorrt, CompositeExplicitAutograd, m) {
  m.impl("execute_engine", execute_engine);
}

} // namespace
} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
