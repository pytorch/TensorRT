#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "Python.h"
#include "core/compiler.h"
#include "core/conversion/conversion.h"
#include "NvInfer.h"
#include "tensorrt_classes.h"
#include "torch/csrc/jit/python/pybind_utils.h"
#include "torch/custom_class.h"
#include "torch/script.h"
#include "torch/torch.h"
#include "utils.h"
using namespace nvinfer1;
using namespace pybind11::literals;
namespace py = pybind11;

namespace trtorch {
namespace pyapi {

  // Use CRTP to share code among several different classes.
  template <typename Derived>
  class pyCalibratorTrampoline : public Derived
  {
  public:
      using Derived::Derived; // Inherit constructors

      int getBatchSize() const noexcept override
      {
          PYBIND11_OVERLOAD_PURE_NAME(int, Derived, "get_batch_size", getBatchSize);
      }

      bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override
      {
          py::gil_scoped_acquire gil{};

          py::function pyGetBatch = tensorrt::utils::getOverload(static_cast<Derived*>(this), "get_batch");
          std::vector<const char*> namesVec(names, names + nbBindings);
          py::object result = pyGetBatch(namesVec);
          // Copy over into the other data structure.
          if (!result.is_none() && result.cast<std::vector<size_t>>().size() != 0)
          {
              std::memcpy(bindings, result.cast<std::vector<size_t>>().data(), nbBindings * sizeof(void*));
              return true;
          }
          return false;
      }

      const void* readCalibrationCache(std::size_t& length) noexcept override
      {
          py::gil_scoped_acquire gil{};

          py::function pyReadCalibrationCache = tensorrt::utils::getOverload(static_cast<Derived*>(this), "read_calibration_cache");
          py::buffer cache = pyReadCalibrationCache();
          if (!cache.is_none())
          {
              py::buffer_info info = cache.request();
              length = info.size * info.itemsize;
              return info.ptr;
          }
          return nullptr;
      }

      void writeCalibrationCache(const void* ptr, std::size_t length) noexcept override
      {
          py::gil_scoped_acquire gil{};

          py::function pyWriteCalibrationCache = tensorrt::utils::getOverload(static_cast<Derived*>(this), "write_calibration_cache");

  # if PYBIND11_VERSION_MAJOR < 2 || PYBIND11_VERSION_MAJOR == 2 && PYBIND11_VERSION_MINOR < 6
          py::buffer_info info{
              const_cast<void*>(ptr),                 /* Pointer to buffer */
              sizeof(uint8_t),                           /* Size of one scalar */
              py::format_descriptor<uint8_t>::format(), /* Python struct-style format descriptor */
              1,                                      /* Number of dimensions */
              { length },                             /* Buffer dimensions */
              { sizeof(uint8_t) }                        /* Strides (in bytes) for each index */
          };
          py::memoryview cache{info};
  #else
          py::memoryview cache{py::memoryview::from_buffer(static_cast<const uint8_t*>(ptr), {length}, {sizeof(uint8_t)} )};
  #endif
          pyWriteCalibrationCache(cache);
      }
  };


  class pyIInt8Calibrator : public pyCalibratorTrampoline<IInt8Calibrator>
  {
  public:
      using Derived = pyCalibratorTrampoline<IInt8Calibrator>;
      using Derived::Derived;

      CalibrationAlgoType getAlgorithm() noexcept override
      {
          PYBIND11_OVERLOAD_PURE_NAME(CalibrationAlgoType, IInt8Calibrator, "get_algorithm", getAlgorithm);
      }
  };

  class pyIInt8LegacyCalibrator : public pyCalibratorTrampoline<IInt8LegacyCalibrator>
  {
  public:
      using Derived = pyCalibratorTrampoline<IInt8LegacyCalibrator>;
      using Derived::Derived;

      double getQuantile() const noexcept override
      {
          PYBIND11_OVERLOAD_PURE_NAME(double, IInt8LegacyCalibrator, "get_quantile", getQuantile);
      }

      double getRegressionCutoff() const noexcept override
      {
          PYBIND11_OVERLOAD_PURE_NAME(double, IInt8LegacyCalibrator, "get_regression_cutoff", getRegressionCutoff);
      }

      const void* readHistogramCache(std::size_t& length) noexcept override
      {
          PYBIND11_OVERLOAD_PURE_NAME(
              const void*, IInt8LegacyCalibrator, "read_histogram_cache", readHistogramCache, length);
      }

      void writeHistogramCache(const void* ptr, std::size_t length) noexcept override
      {
          PYBIND11_OVERLOAD_PURE_NAME(
              void, IInt8LegacyCalibrator, "write_histogram_cache", writeHistogramCache, ptr, length);
      }
  };

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

  py::enum_<CalibrationAlgoType>(m, "CalibrationAlgoType", "Type of calibration algorithm")
      .value("LEGACY_CALIBRATION", CalibrationAlgoType::kLEGACY_CALIBRATION)
      .value("ENTROPY_CALIBRATION", CalibrationAlgoType::kENTROPY_CALIBRATION)
      .value("ENTROPY_CALIBRATION_2", CalibrationAlgoType::kENTROPY_CALIBRATION_2)
      .value("MINMAX_CALIBRATION", CalibrationAlgoType::kMINMAX_CALIBRATION);

  py::class_<IInt8Calibrator, pyIInt8Calibrator>(m, "IInt8Calibrator", "Int8 Calibrator base class")
      .def(py::init_alias<>()) // Always initialize trampoline class.
      .def("get_batch_size", &IInt8Calibrator::getBatchSize, "Get batch size")
      .def("get_algorithm", &IInt8Calibrator::getAlgorithm, "Get algorithm");
      // For documentation purposes only
      // .def("get_batch", docGetBatch<IInt8Calibrator>, "names"_a, "Get Batch")
      // .def("read_calibration_cache", docReadCalibrationCache<IInt8Calibrator>,
      //     "Read calibration cache")
      // .def("write_calibration_cache", docWriteCalibrationCache<IInt8Calibrator>, "cache"_a,
      //     "Write calibration cache");
  py::class_<IInt8LegacyCalibrator, IInt8Calibrator, pyIInt8LegacyCalibrator>(m, "IInt8LegacyCalibrator", "Int8 Legacy Calibrator class")
      .def(py::init_alias<>()) // Always initialize trampoline class.
      .def("get_batch_size", &IInt8LegacyCalibrator::getBatchSize, "Get batch size")
      .def("get_algorithm", &IInt8LegacyCalibrator::getAlgorithm, "Get algorithm");
  //
  py::class_<IInt8EntropyCalibrator, IInt8Calibrator, pyCalibratorTrampoline<IInt8EntropyCalibrator>>(m, "IInt8EntropyCalibrator", "Int8 Entropy Calibrator class")
      .def(py::init_alias<>()) // Always initialize trampoline class.
      .def("get_batch_size", &IInt8EntropyCalibrator::getBatchSize, "Get batch size")
      .def("get_algorithm", &IInt8EntropyCalibrator::getAlgorithm, "Get algorithm");
  //
  py::class_<IInt8EntropyCalibrator2, IInt8Calibrator, pyCalibratorTrampoline<IInt8EntropyCalibrator2>>(m, "IInt8EntropyCalibrator2", "Int8 Entropy Calibrator2 class")
      .def(py::init_alias<>()) // Always initialize trampoline class.
      .def("get_batch_size", &IInt8EntropyCalibrator2::getBatchSize, "Get batch size")
      .def("get_algorithm", &IInt8EntropyCalibrator2::getAlgorithm, "Get algorithm");
  // ;
  //
  py::class_<IInt8MinMaxCalibrator, IInt8Calibrator, pyCalibratorTrampoline<IInt8MinMaxCalibrator>>(m, "IInt8MinMaxCalibrator", "Int8 MinMax Calibrator class")
      .def(py::init_alias<>()) // Always initialize trampoline class.
      .def("get_batch_size", &IInt8MinMaxCalibrator::getBatchSize, "Get batch size")
      .def("get_algorithm", &IInt8MinMaxCalibrator::getAlgorithm, "Get algorithm");
  // ;
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
