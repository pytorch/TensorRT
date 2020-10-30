#pragma once

#include "core/compiler.h"
#include "core/conversion/conversion.h"
#include "torch/torch.h"
#include "torch/script.h"
#include "torch/custom_class.h"

namespace trtorch {
namespace pyapi {

#define ADD_FIELD_GET_SET(field_name, type)             \
  void set_##field_name(type val) {field_name = val;}   \
  type get_##field_name() {return field_name;}

struct InputRange : torch::CustomClassHolder {
  std::vector<int64_t> min;
  std::vector<int64_t> opt;
  std::vector<int64_t> max;

  core::conversion::InputRange toInternalInputRange() {
    return core::conversion::InputRange(min, opt, max);
  }

  ADD_FIELD_GET_SET(min, std::vector<int64_t>);
  ADD_FIELD_GET_SET(opt, std::vector<int64_t>);
  ADD_FIELD_GET_SET(max, std::vector<int64_t>);
};

std::string to_str(InputRange& value);


enum class DataType : int8_t {
  kFloat,
  kHalf,
  kChar,
};

std::string to_str(DataType value);
nvinfer1::DataType toTRTDataType(DataType value);

enum DeviceType : int8_t {
  kGPU,
  kDLA,
};

std::string to_str(DeviceType value);
nvinfer1::DeviceType toTRTDeviceType(DeviceType value);

enum class EngineCapability : int8_t {
    kDEFAULT,
    kSAFE_GPU,
    kSAFE_DLA,
};

std::string to_str(EngineCapability value);
nvinfer1::EngineCapability toTRTEngineCapability(EngineCapability value);

// TODO: Make this error message more informative
#define ADD_ENUM_GET_SET(field_name, type, max_val)                \
  void set_##field_name(int64_t val) {                             \
    TRTORCH_CHECK(val < max_val, "Invalid enum value for field");  \
    field_name = static_cast<type>(val);                           \
  }                                                                \
  int64_t get_##field_name() {return static_cast<int64_t>(field_name);}

struct CompileSpec : torch::CustomClassHolder {
  core::CompileSpec toInternalCompileSpec();
  std::string stringify();
  void appendInputRange(const c10::intrusive_ptr<InputRange>& ir) {
    input_ranges.push_back(*ir);
  }

  ADD_ENUM_GET_SET(op_precision, DataType, 3);
  ADD_FIELD_GET_SET(refit, bool);
  ADD_FIELD_GET_SET(debug, bool);
  ADD_FIELD_GET_SET(strict_types, bool);
  ADD_FIELD_GET_SET(allow_gpu_fallback, bool);
  ADD_ENUM_GET_SET(device, DeviceType, 2);
  ADD_ENUM_GET_SET(capability, EngineCapability, 3);
  ADD_FIELD_GET_SET(num_min_timing_iters, int64_t);
  ADD_FIELD_GET_SET(num_avg_timing_iters, int64_t);
  ADD_FIELD_GET_SET(workspace_size, int64_t);
  ADD_FIELD_GET_SET(max_batch_size, int64_t);

  std::vector<InputRange> input_ranges;
  DataType op_precision = DataType::kFloat;
  bool refit = false;
  bool debug = false;
  bool strict_types = false;
  bool allow_gpu_fallback = true;
  DeviceType device = DeviceType::kGPU;
  EngineCapability capability = EngineCapability::kDEFAULT;
  int64_t num_min_timing_iters = 2;
  int64_t num_avg_timing_iters = 1;
  int64_t workspace_size = 0;
  int64_t max_batch_size = 0;
};

} // namespace pyapi
} // namespace trtorch