#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "core/compiler.h"
#include "core/conversion/conversion.h"
#include "torch/torch.h"
#include "torch/script.h"
#include "torch/csrc/jit/python/pybind_utils.h"
#include "Python.h"

namespace py = pybind11;

namespace trtorch {
namespace pyapi {

struct InputRange {
  std::vector<int64_t> min;
  std::vector<int64_t> opt;
  std::vector<int64_t> max;

  core::conversion::InputRange toInternalInputRange() {
    return core::conversion::InputRange(min, opt, max);
  }
};

enum class DataType : int8_t {
  kFloat,
  kHalf,
  kChar,
};

nvinfer1::DataType toTRTDataType(DataType value) {
  switch (value) {
  case DataType::kChar:
    return nvinfer1::DataType::kINT8;
  case DataType::kHalf:
    return nvinfer1::DataType::kHALF;
  case DataType::kFloat:
  default:
    return nvinfer1::DataType::kFLOAT;
  }
}

enum DeviceType : int8_t {
  kGPU,
  kDLA,
};

nvinfer1::DeviceType toTRTDeviceType(DeviceType value) {
  switch (value) {
  case DeviceType::kDLA:
    return nvinfer1::DeviceType::kDLA;
  case DeviceType::kGPU:
  default:
    return nvinfer1::DeviceType::kDLA;
  }
}

enum class EngineCapability : int8_t {
    kDEFAULT,
    kSAFE_GPU,
    kSAFE_DLA,
};

nvinfer1::EngineCapability toTRTEngineCapability(EngineCapability value) {
  switch (value) {
  case EngineCapability::kSAFE_DLA:
    return nvinfer1::EngineCapability::kSAFE_DLA;
  case EngineCapability::kSAFE_GPU:
    return nvinfer1::EngineCapability::kSAFE_GPU;
  case EngineCapability::kDEFAULT:
  default:
    return nvinfer1::EngineCapability::kDEFAULT;
  }
}

struct ExtraInfo {

  core::ExtraInfo toInternalExtraInfo() {
    auto info = core::ExtraInfo(input_ranges);
    info.convert_info.engine_settings.op_precision = toTRTDataType(op_precision);
    info.convert_info.engine_settings.refit = refit;
    info.convert_info.engine_settings.debug = debug;
    info.convert_info.engine_settings.strict_types = strict_types;
    info.convert_info.engine_settings.allow_gpu_fallback = allow_gpu_fallback;
    info.convert_info.engine_settings.device = toTRTDeviceType(device);
    info.convert_info.engine_settings.capability = toTRTEngineCapability(capability);
    info.convert_info.engine_settings.num_min_timing_iters = num_min_timing_iters;
    info.convert_info.engine_settings.num_avg_timing_iters = num_avg_timing_iters;
    info.convert_info.engine_settings.workspace_size = workspace_size;
    info.convert_info.engine_settings.max_batch_size = max_batch_size;
    return info;
  }

  std::vector<core::conversion::InputRange> input_ranges;
  DataType op_precision = DataType::kFloat;
  bool refit = false;
  bool debug = false;
  bool strict_types = false;
  bool allow_gpu_fallback = true;
  DeviceType device = DeviceType::kGPU;
  EngineCapability capability = EngineCapability::kDEFAULT;
  uint64_t num_min_timing_iters = 2;
  uint64_t num_avg_timing_iters = 1;
  uint64_t workspace_size = 0;
  uint64_t max_batch_size = 0;
};


torch::jit::Module CompileGraph(const torch::jit::Module& mod, ExtraInfo& info) {
  py::gil_scoped_acquire gil;
  auto trt_mod = core::CompileGraph(mod, info.toInternalExtraInfo());
  return trt_mod;
}

std::string ConvertGraphToTRTEngine(const torch::jit::Module& mod, const std::string& method_name, ExtraInfo& info) {
  py::gil_scoped_acquire gil;
  auto trt_engine = core::ConvertGraphToTRTEngine(mod, method_name, info.toInternalExtraInfo());
  return trt_engine;
}

bool CheckMethodOperatorSupport(const torch::jit::Module& module, const std::string& method_name) {
  return core::CheckMethodOperatorSupport(module, method_name);
}

void test(torch::jit::Module& mod, torch::Tensor data) {
  std::cout << mod.forward({data}) << std::endl;
}

std::string get_build_info() {
  auto info = core::util::get_build_info();
  return info;
}

PYBIND11_MODULE(_C, m) {
  py::class_<InputRange>(m, "InputRange")
    .def(py::init<>())
    .def_readwrite("min", &InputRange::min)
    .def_readwrite("opt", &InputRange::opt)
    .def_readwrite("max", &InputRange::max)
    .def("_to_internal_input_range", &InputRange::toInternalInputRange);

  //py::class_<core::conversion::InputRange>(m, "_InternalInputRange")
  //    .def(py::init<>());

  py::enum_<DataType>(m, "dtype")
    .value("float",   DataType::kFloat)
    .value("float32", DataType::kFloat)
    .value("half",    DataType::kHalf)
    .value("float16", DataType::kHalf)
    .value("int8",    DataType::kChar)
    .export_values();

  py::enum_<DeviceType>(m, "DeviceType")
    .value("gpu", DeviceType::kGPU)
    .value("dla", DeviceType::kDLA)
    .export_values();

  py::enum_<EngineCapability>(m, "EngineCapability")
    .value("safe_gpu", EngineCapability::kSAFE_GPU)
    .value("safe_dla", EngineCapability::kSAFE_DLA)
    .value("default",  EngineCapability::kDEFAULT);

  py::class_<ExtraInfo>(m, "_ExtraInfo")
    .def(py::init<>())
    .def_readwrite("input_ranges",         &ExtraInfo::input_ranges)
    .def_readwrite("op_precision",         &ExtraInfo::op_precision)
    .def_readwrite("refit",                &ExtraInfo::refit)
    .def_readwrite("debug",                &ExtraInfo::debug)
    .def_readwrite("strict_types",         &ExtraInfo::strict_types)
    .def_readwrite("allow_gpu_fallback",   &ExtraInfo::allow_gpu_fallback)
    .def_readwrite("device",               &ExtraInfo::device)
    .def_readwrite("capability",           &ExtraInfo::capability)
    .def_readwrite("num_min_timing_iters", &ExtraInfo::num_min_timing_iters)
    .def_readwrite("num_avg_timing_iters", &ExtraInfo::num_avg_timing_iters)
    .def_readwrite("workspace_size",       &ExtraInfo::workspace_size)
    .def_readwrite("max_batch_size",       &ExtraInfo::max_batch_size);

  m.doc() = "TRTorch Internal C Bindings: Ahead of Time compilation for PyTorch JIT. A tool to convert PyTorch JIT to TensorRT";
  m.def("_compile_graph",               &trtorch::pyapi::CompileGraph, "Ingest a PyTorch JIT module and convert supported subgraphs to TensorRT engines, returns a JIT module with the engines embedded");
  m.def("_convert_graph_to_trt_engine", &trtorch::pyapi::ConvertGraphToTRTEngine, "Given a PyTorch JIT Module, convert forward into a TensorRT engine and return a serialized engine");
  m.def("_check_method_op_support",     &trtorch::pyapi::CheckMethodOperatorSupport, "Takes a module and a method name and checks if the method graph contains purely convertable operators");
  m.def("_get_build_info",              &get_build_info, "Returns build info about the compiler as a string");
  m.def("_test", &test);
}

// namespace logging {
// PYBIND11_MODULE(logging, m) {
//     m.attr("__name__") = "trtorch.logging";
//     m.def("get_logging_prefix", &trtorch::logging::get_logging_prefix, "Get the current prefix for the logging output");
//     m.def("set_logging_prefix", &trtorch::logging::set_logging_prefix, "Set the logging prefix for logging output");
//     m.def("get_reportable_log_level", &trtorch::logging::get_reportable_log_level, "Get the current log level");
//     m.def("set_reportable_log_level", &trtorch::logging::set_reportable_log_level, "Set the level required to be met for a log message to be printed");
//     m.def("get_is_colored_output_on", &trtorch::logging::get_is_colored_output_on, "Get if the logging output will be colored");
//     m.def("set_is_colored_output_on", &trtorch::logging::set_is_colored_output_on, "Set if the logging output should be colored");
//     m.def("log", &trtorch::logging::log, "Add a message to the logger");
//     py::enum_<trtorch::logging::Level>(m, "Level", py::arithmetic())
//         .value("INTERNAL_ERROR", trtorch::logging::Level::kINTERNAL_ERROR)
//         .value("ERROR", trtorch::logging::Level::kERROR)
//         .value("WARNING", trtorch::logging::Level::kWARNING)
//         .value("INFO", trtorch::logging::Level::kINFO)
//         .value("DEBUG", trtorch::logging::Level::kDEBUG)
//         .export_values();
// }
//} // namespace logging
} // namespace py
} // namespace trtorch
