#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "ATen/core/jit_type.h"
#include "Python.h"
#include "core/compiler.h"
#include "core/conversion/conversion.h"
#include "tensorrt_classes.h"
#include "torch/csrc/jit/python/pybind_utils.h"
#include "torch/custom_class.h"
#include "torch/script.h"
#include "torch/torch.h"
#include "util.h"

namespace py = pybind11;

namespace torch_tensorrt {
namespace pyapi {

template <typename Derived>
class pyCalibratorTrampoline : public Derived {
 public:
  using Derived::Derived; // Inherit constructors

  int getBatchSize() const noexcept override {
    try {
      PYBIND11_OVERLOAD_PURE_NAME(int, Derived, "get_batch_size", getBatchSize);
    } catch (std::exception const& e) {
      LOG_ERROR("Exception caught in get_batch_size" + std::string(e.what()));
    } catch (...) {
      LOG_ERROR("Exception caught in get_batch_size");
    }
    return -1;
  }

  bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override {
    py::gil_scoped_acquire gil{};

    py::function pyGetBatch = torch_tensorrt::pyapi::util::getOverload(static_cast<Derived*>(this), "get_batch");
    std::vector<const char*> namesVec(names, names + nbBindings);
    py::object result = pyGetBatch(namesVec);
    // Copy over into the other data structure.
    if (!result.is_none() && result.cast<std::vector<size_t>>().size() != 0) {
      std::memcpy(bindings, result.cast<std::vector<size_t>>().data(), nbBindings * sizeof(void*));
      return true;
    }
    return false;
  }

  const void* readCalibrationCache(std::size_t& length) noexcept override {
    py::gil_scoped_acquire gil{};

    py::function pyReadCalibrationCache =
        torch_tensorrt::pyapi::util::getOverload(static_cast<Derived*>(this), "read_calibration_cache");
    py::buffer cache = pyReadCalibrationCache();
    if (!cache.is_none()) {
      py::buffer_info info = cache.request();
      length = info.size * info.itemsize;
      return info.ptr;
    }
    return nullptr;
  }

  void writeCalibrationCache(const void* ptr, std::size_t length) noexcept override {
    py::gil_scoped_acquire gil{};

    py::function pyWriteCalibrationCache =
        torch_tensorrt::pyapi::util::getOverload(static_cast<Derived*>(this), "write_calibration_cache");

    py::memoryview cache{py::memoryview::from_buffer(static_cast<const uint8_t*>(ptr), {length}, {sizeof(uint8_t)})};
    pyWriteCalibrationCache(cache);
  }
};

class pyIInt8Calibrator : public pyCalibratorTrampoline<nvinfer1::IInt8Calibrator> {
 public:
  using Derived = pyCalibratorTrampoline<nvinfer1::IInt8Calibrator>;
  using Derived::Derived;

  nvinfer1::CalibrationAlgoType getAlgorithm() noexcept override {
    try {
      PYBIND11_OVERLOAD_PURE_NAME(
          nvinfer1::CalibrationAlgoType, nvinfer1::IInt8Calibrator, "get_algorithm", getAlgorithm);
    } catch (std::exception const& e) {
      LOG_ERROR("Exception caught in get_algorithm: " + std::string(e.what()));
    } catch (...) {
      LOG_ERROR("Exception caught in get_algorithm");
    }
    return {};
  }
};

class pyIInt8LegacyCalibrator : public pyCalibratorTrampoline<nvinfer1::IInt8LegacyCalibrator> {
 public:
  using Derived = pyCalibratorTrampoline<nvinfer1::IInt8LegacyCalibrator>;
  using Derived::Derived;

  double getQuantile() const noexcept override {
    try {
      PYBIND11_OVERLOAD_PURE_NAME(double, nvinfer1::IInt8LegacyCalibrator, "get_quantile", getQuantile);
    } catch (std::exception const& e) {
      LOG_ERROR("Exception caught in get_quantile: " + std::string(e.what()));
    } catch (...) {
      LOG_ERROR("Exception caught in get_quantile");
    }
    return -1.0;
  }

  double getRegressionCutoff() const noexcept override {
    try {
      PYBIND11_OVERLOAD_PURE_NAME(
          double, nvinfer1::IInt8LegacyCalibrator, "get_regression_cutoff", getRegressionCutoff);
    } catch (std::exception const& e) {
      LOG_ERROR("Exception caught in get_regression_cutoff: " + std::string(e.what()));
    } catch (...) {
      LOG_ERROR("Exception caught in get_regression_cutoff");
    }
    return -1.0;
  }

  const void* readHistogramCache(std::size_t& length) noexcept override {
    try {
      PYBIND11_OVERLOAD_PURE_NAME(
          const char*, nvinfer1::IInt8LegacyCalibrator, "read_histogram_cache", readHistogramCache, length);
    } catch (std::exception const& e) {
      LOG_ERROR("Exception caught in read_histogram_cache" + std::string(e.what()));
    } catch (...) {
      LOG_ERROR("Exception caught in read_histogram_cache");
    }
    return {};
  }

  void writeHistogramCache(const void* ptr, std::size_t length) noexcept override {
    try {
      PYBIND11_OVERLOAD_PURE_NAME(
          void, nvinfer1::IInt8LegacyCalibrator, "write_histogram_cache", writeHistogramCache, ptr, length);
    } catch (std::exception const& e) {
      LOG_ERROR("Exception caught in write_histogram_cache" + std::string(e.what()));
    } catch (...) {
      LOG_ERROR("Exception caught in write_histogram_cache");
    }
  }
};

void set_device(const int device_id) {
  core::set_device(device_id);
}

Device get_current_device() {
  return Device(core::runtime::get_current_device());
}

torch::jit::Module CompileGraph(const torch::jit::Module& mod, CompileSpec& info) {
  py::gil_scoped_acquire gil;
  auto trt_mod = core::CompileGraph(mod, info.toInternalCompileSpec());
  return trt_mod;
}

py::bytes ConvertGraphToTRTEngine(const torch::jit::Module& mod, const std::string& method_name, CompileSpec& info) {
  py::gil_scoped_acquire gil;
  auto trt_engine = core::ConvertGraphToTRTEngine(
      mod, method_name, info.toInternalCompileSpec(/*bool converting_to_trt_engine=*/true));
  return py::bytes(trt_engine);
}

bool CheckMethodOperatorSupport(const torch::jit::Module& module, const std::string& method_name) {
  return core::CheckMethodOperatorSupport(module, method_name);
}

torch::jit::Module EmbedEngineInNewModule(
    const py::bytes& engine,
    Device& device,
    const std::vector<std::string>& input_binding_names,
    const std::vector<std::string>& output_binding_names) {
  return core::EmbedEngineInNewModule(engine, device.toInternalRTDevice(), input_binding_names, output_binding_names);
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
  py::class_<Input>(m, "Input")
      .def(py::init<>())
      .def("__str__", &torch_tensorrt::pyapi::Input::to_str)
      .def_readwrite("min", &Input::min)
      .def_readwrite("opt", &Input::opt)
      .def_readwrite("max", &Input::max)
      .def_readwrite("input_is_dynamic", &Input::input_is_dynamic)
      .def_readwrite("_explicit_set_dtype", &Input::explicit_set_dtype)
      .def_readwrite("dtype", &Input::dtype)
      .def_readwrite("tensor_domain", &Input::tensor_domain)
      .def_readwrite("format", &Input::format);

  py::class_<InputSignature>(m, "InputSignature")
      .def(pybind11::init([](py::object py_obj) {
        InputSignature input_signature;
        input_signature.signature_ivalue =
            torch::jit::toIValue(std::move(py_obj), c10::PyObjectType::get(), c10::nullopt);
        return input_signature;
      }))
      .def("__str__", &InputSignature::to_str)
      .def_readwrite("_signature_ivalue", &InputSignature::signature_ivalue);

  py::enum_<DataType>(m, "dtype", "Enum to specifiy operating precision for engine execution")
      .value("float", DataType::kFloat, "32 bit floating point number")
      .value("float32", DataType::kFloat, "32 bit floating point number")
      .value("half", DataType::kHalf, "16 bit floating point number")
      .value("float16", DataType::kHalf, "16 bit floating point number")
      .value("int8", DataType::kChar, "8 bit integer number")
      .value("int32", DataType::kInt32, "32 bit integer number")
      .value("long", DataType::kLong, "64 bit integer number")
      .value("int64", DataType::kLong, "64 bit integer number")
      .value("bool", DataType::kBool, "Boolean value")
      .value("unknown", DataType::kUnknown, "Unknown data type")
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

  py::enum_<TensorFormat>(m, "TensorFormat", "Enum to specifiy the memory layout of tensors")
      .value("contiguous", TensorFormat::kContiguous, "Contiguous memory layout (NCHW / Linear)")
      .value("channels_last", TensorFormat::kChannelsLast, "Channels last memory layout (NHWC)")
      .export_values();

  py::enum_<nvinfer1::CalibrationAlgoType>(m, "CalibrationAlgo", py::module_local(), "Type of calibration algorithm")
      .value("LEGACY_CALIBRATION", nvinfer1::CalibrationAlgoType::kLEGACY_CALIBRATION)
      .value("ENTROPY_CALIBRATION", nvinfer1::CalibrationAlgoType::kENTROPY_CALIBRATION)
      .value("ENTROPY_CALIBRATION_2", nvinfer1::CalibrationAlgoType::kENTROPY_CALIBRATION_2)
      .value("MINMAX_CALIBRATION", nvinfer1::CalibrationAlgoType::kMINMAX_CALIBRATION);

  py::class_<nvinfer1::IInt8Calibrator, pyIInt8Calibrator>(
      m, "IInt8Calibrator", py::module_local(), "Int8 Calibrator base class")
      .def(py::init_alias<>()) // Always initialize trampoline class.
      .def("get_batch_size", &nvinfer1::IInt8Calibrator::getBatchSize, "Get batch size")
      .def("get_algorithm", &nvinfer1::IInt8Calibrator::getAlgorithm, "Get algorithm");

  py::class_<nvinfer1::IInt8LegacyCalibrator, nvinfer1::IInt8Calibrator, pyIInt8LegacyCalibrator>(
      m, "IInt8LegacyCalibrator", py::module_local(), "Int8 Legacy Calibrator class")
      .def(py::init_alias<>()) // Always initialize trampoline class.
      .def("get_batch_size", &nvinfer1::IInt8LegacyCalibrator::getBatchSize, "Get batch size")
      .def("get_algorithm", &nvinfer1::IInt8LegacyCalibrator::getAlgorithm, "Get algorithm");

  py::class_<
      nvinfer1::IInt8EntropyCalibrator,
      nvinfer1::IInt8Calibrator,
      pyCalibratorTrampoline<nvinfer1::IInt8EntropyCalibrator>>(
      m, "IInt8EntropyCalibrator", py::module_local(), "Int8 Entropy Calibrator class")
      .def(py::init_alias<>()) // Always initialize trampoline class.
      .def("get_batch_size", &nvinfer1::IInt8EntropyCalibrator::getBatchSize, "Get batch size")
      .def("get_algorithm", &nvinfer1::IInt8EntropyCalibrator::getAlgorithm, "Get algorithm");

  py::class_<
      nvinfer1::IInt8EntropyCalibrator2,
      nvinfer1::IInt8Calibrator,
      pyCalibratorTrampoline<nvinfer1::IInt8EntropyCalibrator2>>(
      m, "IInt8EntropyCalibrator2", py::module_local(), "Int8 Entropy Calibrator2 class")
      .def(py::init_alias<>()) // Always initialize trampoline class.
      .def("get_batch_size", &nvinfer1::IInt8EntropyCalibrator2::getBatchSize, "Get batch size")
      .def("get_algorithm", &nvinfer1::IInt8EntropyCalibrator2::getAlgorithm, "Get algorithm");

  py::class_<
      nvinfer1::IInt8MinMaxCalibrator,
      nvinfer1::IInt8Calibrator,
      pyCalibratorTrampoline<nvinfer1::IInt8MinMaxCalibrator>>(
      m, "IInt8MinMaxCalibrator", py::module_local(), "Int8 MinMax Calibrator class")
      .def(py::init_alias<>()) // Always initialize trampoline class.
      .def("get_batch_size", &nvinfer1::IInt8MinMaxCalibrator::getBatchSize, "Get batch size")
      .def("get_algorithm", &nvinfer1::IInt8MinMaxCalibrator::getAlgorithm, "Get algorithm");

  py::class_<Device>(m, "Device")
      .def(py::init<>())
      .def("__str__", &torch_tensorrt::pyapi::Device::to_str)
      .def("_to_serialized_rt_device", &torch_tensorrt::pyapi::Device::toSerializedRTDevice)
      .def_readwrite("device_type", &Device::device_type)
      .def_readwrite("gpu_id", &Device::gpu_id)
      .def_readwrite("dla_core", &Device::dla_core)
      .def_readwrite("allow_gpu_fallback", &Device::allow_gpu_fallback);

  m.doc() = "Torch-TensorRT Internal C Bindings: A tool to convert PyTorch to TensorRT";

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
  m.def("set_device", &torch_tensorrt::pyapi::set_device, "Set CUDA device id");
  m.def("_get_current_device", &torch_tensorrt::pyapi::get_current_device, "Get the current active CUDA device");

  py::enum_<core::util::logging::LogLevel>(m, "LogLevel", py::arithmetic())
      .value("INTERNAL_ERROR", core::util::logging::LogLevel::kINTERNAL_ERROR)
      .value("ERROR", core::util::logging::LogLevel::kERROR)
      .value("WARNING", core::util::logging::LogLevel::kWARNING)
      .value("INFO", core::util::logging::LogLevel::kINFO)
      .value("DEBUG", core::util::logging::LogLevel::kDEBUG)
      .value("GRAPH", core::util::logging::LogLevel::kGRAPH)
      .export_values();

  py::module rt_sub_mod = m.def_submodule("rt");
  rt_sub_mod.attr("ABI_VERSION") = std::string(core::runtime::ABI_VERSION);

  py::module ts_sub_mod = m.def_submodule("ts");
  py::class_<CompileSpec>(ts_sub_mod, "CompileSpec")
      .def(py::init<>())
      .def("__str__", &torch_tensorrt::pyapi::CompileSpec::stringify)
      .def("_get_calibrator_handle", &CompileSpec::getPTQCalibratorHandle, "[Internal] gets a handle from a calibrator")
      .def_readwrite("inputs", &CompileSpec::inputs)
      .def_readwrite("input_signature", &CompileSpec::input_signature)
      .def_readwrite("enabled_precisions", &CompileSpec::enabled_precisions)
      .def_readwrite("ptq_calibrator", &CompileSpec::ptq_calibrator)
      .def_readwrite("refit", &CompileSpec::refit)
      .def_readwrite("sparse_weights", &CompileSpec::sparse_weights)
      .def_readwrite("disable_tf32", &CompileSpec::disable_tf32)
      .def_readwrite("debug", &CompileSpec::debug)
      .def_readwrite("device", &CompileSpec::device)
      .def_readwrite("capability", &CompileSpec::capability)
      .def_readwrite("num_avg_timing_iters", &CompileSpec::num_avg_timing_iters)
      .def_readwrite("workspace_size", &CompileSpec::workspace_size)
      .def_readwrite("dla_sram_size", &CompileSpec::dla_sram_size)
      .def_readwrite("dla_local_dram_size", &CompileSpec::dla_local_dram_size)
      .def_readwrite("dla_global_dram_size", &CompileSpec::dla_global_dram_size)
      .def_readwrite("torch_fallback", &CompileSpec::torch_fallback)
      .def_readwrite("truncate_long_and_double", &CompileSpec::truncate_long_and_double)
      .def_readwrite("allow_shape_tensors", &CompileSpec::allow_shape_tensors);

  py::class_<TorchFallback>(ts_sub_mod, "TorchFallback")
      .def(py::init<>())
      .def("__str__", &torch_tensorrt::pyapi::TorchFallback::to_str)
      .def_readwrite("enabled", &TorchFallback::enabled)
      .def_readwrite("min_block_size", &TorchFallback::min_block_size)
      .def_readwrite("forced_fallback_operators", &TorchFallback::forced_fallback_operators)
      .def_readwrite("forced_fallback_modules", &TorchFallback::forced_fallback_modules);

  ts_sub_mod.def(
      "compile_graph",
      &torch_tensorrt::pyapi::CompileGraph,
      "Ingest a PyTorch JIT module and convert supported subgraphs to TensorRT engines, returns a JIT module with the engines embedded");
  ts_sub_mod.def(
      "convert_graph_to_trt_engine",
      &torch_tensorrt::pyapi::ConvertGraphToTRTEngine,
      "Given a PyTorch JIT Module, convert forward into a TensorRT engine and return a serialized engine");
  ts_sub_mod.def(
      "check_method_op_support",
      &torch_tensorrt::pyapi::CheckMethodOperatorSupport,
      "Takes a module and a method name and checks if the method graph contains purely convertable operators");
  ts_sub_mod.def(
      "embed_engine_in_new_module",
      &torch_tensorrt::pyapi::EmbedEngineInNewModule,
      "Takes a serialized TensorRT engine and compile spec. Wraps it in the forward method of a new TorchScript module");

  ts_sub_mod.doc() =
      "Torch-TensorRT TorchScript Compiler Internal C Bindings: AOT Compilation for PyTorch JIT to TensorRT";
}

} // namespace pyapi
} // namespace torch_tensorrt
