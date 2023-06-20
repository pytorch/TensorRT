#pragma once

#include "core/compiler.h"
#include "core/conversion/conversion.h"
#include "torch/custom_class.h"
#include "torch/script.h"
#include "torch/torch.h"

namespace torch_tensorrt {
namespace pyapi {

#define ADD_FIELD_GET_SET(field_name, type) \
  void set_##field_name(type val) {         \
    field_name = val;                       \
  }                                         \
  type get_##field_name() {                 \
    return field_name;                      \
  }

// TODO: Make this error message more informative
#define ADD_ENUM_GET_SET(field_name, type, max_val)                             \
  void set_##field_name(int64_t val) {                                          \
    TORCHTRT_CHECK(val >= 0 && val <= max_val, "Invalid enum value for field"); \
    field_name = static_cast<type>(val);                                        \
  }                                                                             \
  int64_t get_##field_name() {                                                  \
    return static_cast<int64_t>(field_name);                                    \
  }

enum class DataType : int8_t { kLong, kFloat, kHalf, kChar, kInt32, kBool, kUnknown };
std::string to_str(DataType value);
nvinfer1::DataType toTRTDataType(DataType value);
at::ScalarType toAtenDataType(DataType value);

enum class TensorFormat : int8_t { kContiguous, kChannelsLast };
std::string to_str(TensorFormat value);
nvinfer1::TensorFormat toTRTTensorFormat(TensorFormat value);

struct Input : torch::CustomClassHolder {
  std::vector<int64_t> min;
  std::vector<int64_t> opt;
  std::vector<int64_t> max;
  std::vector<double> tensor_domain;

  bool input_is_dynamic;
  bool explicit_set_dtype;
  DataType dtype;
  TensorFormat format;

  ADD_FIELD_GET_SET(min, std::vector<int64_t>);
  ADD_FIELD_GET_SET(opt, std::vector<int64_t>);
  ADD_FIELD_GET_SET(max, std::vector<int64_t>);
  ADD_FIELD_GET_SET(tensor_domain, std::vector<double>);
  ADD_FIELD_GET_SET(input_is_dynamic, bool);
  ADD_FIELD_GET_SET(explicit_set_dtype, bool);
  ADD_ENUM_GET_SET(dtype, DataType, static_cast<int64_t>(DataType::kUnknown));
  ADD_ENUM_GET_SET(format, TensorFormat, static_cast<int64_t>(TensorFormat::kContiguous));

  core::ir::Input toInternalInput();
  std::string to_str();
};

struct InputSignature : torch::CustomClassHolder {
  torch::jit::IValue signature_ivalue; // nested Input, full input spec
  ADD_FIELD_GET_SET(signature_ivalue, torch::jit::IValue);
  std::string to_str();
};

enum DeviceType : int8_t {
  kGPU,
  kDLA,
};

struct Device : torch::CustomClassHolder {
  DeviceType device_type;
  int64_t gpu_id;
  int64_t dla_core;
  bool allow_gpu_fallback;
  Device()
      : device_type(DeviceType::kGPU), // device_type
        gpu_id(0), // gpu_id
        dla_core(0), // dla_core
        allow_gpu_fallback(false) // allow_gpu_fallback
  {}

  Device(const core::runtime::RTDevice& internal_dev);

  ADD_ENUM_GET_SET(device_type, DeviceType, static_cast<int64_t>(DeviceType::kDLA));
  ADD_FIELD_GET_SET(gpu_id, int64_t);
  ADD_FIELD_GET_SET(dla_core, int64_t);
  ADD_FIELD_GET_SET(allow_gpu_fallback, bool);

  core::runtime::RTDevice toInternalRTDevice();
  std::string toSerializedRTDevice();
  std::string to_str();
};

std::string to_str(DeviceType value);
nvinfer1::DeviceType toTRTDeviceType(DeviceType value);

struct TorchFallback : torch::CustomClassHolder {
  bool enabled;
  int64_t min_block_size;
  std::vector<std::string> forced_fallback_operators;
  std::vector<std::string> forced_fallback_modules;
  TorchFallback() : enabled(false), min_block_size(1) {}

  ADD_FIELD_GET_SET(enabled, bool);
  ADD_FIELD_GET_SET(min_block_size, int64_t);
  ADD_FIELD_GET_SET(forced_fallback_operators, std::vector<std::string>);
  ADD_FIELD_GET_SET(forced_fallback_modules, std::vector<std::string>);

  std::string to_str();
};

enum class EngineCapability : int8_t {
  kDEFAULT,
  kSAFE_GPU,
  kSAFE_DLA,
};

std::string to_str(EngineCapability value);
nvinfer1::EngineCapability toTRTEngineCapability(EngineCapability value);

struct CompileSpec : torch::CustomClassHolder {
  core::CompileSpec toInternalCompileSpec(bool converting_to_trt_engine = false);
  std::string stringify();
  void appendInput(const c10::intrusive_ptr<Input>& ir) {
    inputs.push_back(*ir);
  }

  void setInputSignature(const c10::intrusive_ptr<InputSignature>& is) {
    input_signature = *is;
  }

  void setPrecisions(const std::vector<int64_t>& precisions_raw) {
    for (auto p : precisions_raw) {
      TORCHTRT_CHECK(p >= 0 && p <= static_cast<int64_t>(DataType::kBool), "Invalid enum value for field");
      enabled_precisions.insert(static_cast<DataType>(p));
    }
  }

  int64_t getPTQCalibratorHandle() {
    return (int64_t)ptq_calibrator;
  }

  void setDeviceIntrusive(const c10::intrusive_ptr<Device>& d) {
    device = *d;
  }

  void setTorchFallbackIntrusive(const c10::intrusive_ptr<TorchFallback>& fb) {
    torch_fallback = *fb;
  }

  void setPTQCalibratorViaHandle(int64_t handle) {
    ptq_calibrator = (nvinfer1::IInt8Calibrator*)handle;
  }

  ADD_FIELD_GET_SET(disable_tf32, bool);
  ADD_FIELD_GET_SET(sparse_weights, bool);
  ADD_FIELD_GET_SET(refit, bool);
  ADD_FIELD_GET_SET(debug, bool);
  ADD_ENUM_GET_SET(capability, EngineCapability, static_cast<int64_t>(EngineCapability::kSAFE_DLA));
  ADD_FIELD_GET_SET(num_avg_timing_iters, int64_t);
  ADD_FIELD_GET_SET(workspace_size, int64_t);
  ADD_FIELD_GET_SET(dla_sram_size, int64_t);
  ADD_FIELD_GET_SET(dla_local_dram_size, int64_t);
  ADD_FIELD_GET_SET(dla_global_dram_size, int64_t);
  ADD_FIELD_GET_SET(truncate_long_and_double, bool);
  ADD_FIELD_GET_SET(allow_shape_tensors, bool);
  ADD_FIELD_GET_SET(device, Device);
  ADD_FIELD_GET_SET(torch_fallback, TorchFallback);
  ADD_FIELD_GET_SET(ptq_calibrator, nvinfer1::IInt8Calibrator*);

  std::vector<Input> inputs;
  InputSignature input_signature;
  nvinfer1::IInt8Calibrator* ptq_calibrator = nullptr;
  std::set<DataType> enabled_precisions = {};
  bool sparse_weights = false;
  bool disable_tf32 = false;
  bool refit = false;
  bool debug = false;
  bool truncate_long_and_double = false;
  bool allow_shape_tensors = false;
  Device device;
  TorchFallback torch_fallback;
  EngineCapability capability = EngineCapability::kDEFAULT;
  int64_t num_avg_timing_iters = 1;
  int64_t workspace_size = 0;
  int64_t dla_sram_size = 1048576;
  int64_t dla_local_dram_size = 1073741824;
  int64_t dla_global_dram_size = 536870912;
};

} // namespace pyapi
} // namespace torch_tensorrt
