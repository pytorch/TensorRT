#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "Python.h"
#include "core/compiler.h"
#include "core/conversion/conversion.h"
#include "tensorrt_classes.h"
#include "torch/csrc/jit/python/pybind_utils.h"
#include "torch/custom_class.h"
#include "torch/script.h"
#include "torch/torch.h"

namespace py = pybind11;

namespace trtorch {
namespace pyapi {

void set_device(const int device_id) {
  core::set_device(device_id);
}

torch::jit::Module CompileGraph(const torch::jit::Module& mod, CompileSpec& info) {
  py::gil_scoped_acquire gil;
  auto trt_mod = core::CompileGraph(mod, info.toInternalCompileSpec());
  return trt_mod;
}

py::bytes ConvertGraphToTRTEngine(const torch::jit::Module& mod, const std::string& method_name, CompileSpec& info) {
  py::gil_scoped_acquire gil;
  auto trt_engine = core::ConvertGraphToTRTEngine(mod, method_name, info.toInternalCompileSpec());
  return py::bytes(trt_engine);
}

bool CheckMethodOperatorSupport(const torch::jit::Module& module, const std::string& method_name) {
  return core::CheckMethodOperatorSupport(module, method_name);
}

std::string get_build_info() {
  auto info = core::util::get_build_info();
  return info;
}

namespace logging {
std::string get_logging_prefix() {
  return core::util::logging::get_logger().get_logging_prefix();
}

void set_logging_prefix(const std::string& prefix) {
  std::string p;
  p.assign(prefix);
  core::util::logging::get_logger().set_logging_prefix(p);
}

void set_reportable_log_level(core::util::logging::LogLevel lvl) {
  core::util::logging::get_logger().set_reportable_log_level(lvl);
}

void set_is_colored_output_on(bool colored_output_on) {
  core::util::logging::get_logger().set_is_colored_output_on(colored_output_on);
}

core::util::logging::LogLevel get_reportable_log_level() {
  return core::util::logging::get_logger().get_reportable_log_level();
}

bool get_is_colored_output_on() {
  return core::util::logging::get_logger().get_is_colored_output_on();
}

void log(core::util::logging::LogLevel lvl, const std::string& msg) {
  std::string m;
  m.assign(msg);
  core::util::logging::get_logger().log(lvl, m);
}
} // namespace logging

PYBIND11_MODULE(_C, m) {
  py::class_<InputRange>(m, "InputRange")
      .def(py::init<>())
      .def_readwrite("min", &InputRange::min)
      .def_readwrite("opt", &InputRange::opt)
      .def_readwrite("max", &InputRange::max);

  py::enum_<DataType>(m, "dtype", "Enum to specifiy operating precision for engine execution")
      .value("float", DataType::kFloat, "32 bit floating point number")
      .value("float32", DataType::kFloat, "32 bit floating point number")
      .value("half", DataType::kHalf, "16 bit floating point number")
      .value("float16", DataType::kHalf, "16 bit floating point number")
      .value("int8", DataType::kChar, "8 bit integer number")
      .export_values();

  py::enum_<DeviceType>(m, "DeviceType", "Enum to specify device kinds to build TensorRT engines for")
      .value("GPU", DeviceType::kGPU, "Specify using GPU to execute TensorRT Engine")
      .value("DLA", DeviceType::kDLA, "Specify using DLA to execute TensorRT Engine (Jetson Only)")
      .export_values();

  py::enum_<EngineCapability>(
      m,
      "EngineCapability",
      "Enum to specify engine capability settings (selections of kernels to meet safety requirements)")
      .value("safe_gpu", EngineCapability::kSAFE_GPU, "Use safety GPU kernels only")
      .value("safe_dla", EngineCapability::kSAFE_DLA, "Use safety DLA kernels only")
      .value("default", EngineCapability::kDEFAULT, "Use default behavior");

  py::class_<CompileSpec>(m, "CompileSpec")
      .def(py::init<>())
      .def_readwrite("input_ranges", &CompileSpec::input_ranges)
      .def_readwrite("op_precision", &CompileSpec::op_precision)
      .def_readwrite("refit", &CompileSpec::refit)
      .def_readwrite("disable_tf32", &CompileSpec::disable_tf32)
      .def_readwrite("debug", &CompileSpec::debug)
      .def_readwrite("strict_types", &CompileSpec::strict_types)
      .def_readwrite("device", &CompileSpec::device)
      .def_readwrite("capability", &CompileSpec::capability)
      .def_readwrite("num_min_timing_iters", &CompileSpec::num_min_timing_iters)
      .def_readwrite("num_avg_timing_iters", &CompileSpec::num_avg_timing_iters)
      .def_readwrite("workspace_size", &CompileSpec::workspace_size)
      .def_readwrite("max_batch_size", &CompileSpec::max_batch_size);

  py::class_<Device>(m, "Device")
      .def(py::init<>())
      .def_readwrite("device_type", &Device::device_type)
      .def_readwrite("gpu_id", &Device::gpu_id)
      .def_readwrite("dla_core", &Device::dla_core)
      .def_readwrite("allow_gpu_fallback", &Device::allow_gpu_fallback);

  m.doc() =
      "TRTorch Internal C Bindings: Ahead of Time compilation for PyTorch JIT. A tool to convert PyTorch JIT to TensorRT";
  m.def(
      "compile_graph",
      &trtorch::pyapi::CompileGraph,
      "Ingest a PyTorch JIT module and convert supported subgraphs to TensorRT engines, returns a JIT module with the engines embedded");
  m.def(
      "convert_graph_to_trt_engine",
      &trtorch::pyapi::ConvertGraphToTRTEngine,
      "Given a PyTorch JIT Module, convert forward into a TensorRT engine and return a serialized engine");
  m.def(
      "check_method_op_support",
      &trtorch::pyapi::CheckMethodOperatorSupport,
      "Takes a module and a method name and checks if the method graph contains purely convertable operators");
  m.def("get_build_info", &get_build_info, "Returns build info about the compiler as a string");

  m.def("_get_logging_prefix", &logging::get_logging_prefix, "Get the current prefix for the logging output");
  m.def("_set_logging_prefix", &logging::set_logging_prefix, "Set the logging prefix for logging output");
  m.def("_get_reportable_log_level", &logging::get_reportable_log_level, "Get the current log level");
  m.def(
      "_set_reportable_log_level",
      &logging::set_reportable_log_level,
      "Set the level required to be met for a log message to be printed");
  m.def("_get_is_colored_output_on", &logging::get_is_colored_output_on, "Get if the logging output will be colored");
  m.def("_set_is_colored_output_on", &logging::set_is_colored_output_on, "Set if the logging output should be colored");
  m.def("_log", &logging::log, "Add a message to the logger");
  m.def("set_device", &trtorch::pyapi::set_device, "Set CUDA device id");

  py::enum_<core::util::logging::LogLevel>(m, "LogLevel", py::arithmetic())
      .value("INTERNAL_ERROR", core::util::logging::LogLevel::kINTERNAL_ERROR)
      .value("ERROR", core::util::logging::LogLevel::kERROR)
      .value("WARNING", core::util::logging::LogLevel::kWARNING)
      .value("INFO", core::util::logging::LogLevel::kINFO)
      .value("DEBUG", core::util::logging::LogLevel::kDEBUG)
      .export_values();
}

} // namespace pyapi
} // namespace trtorch
