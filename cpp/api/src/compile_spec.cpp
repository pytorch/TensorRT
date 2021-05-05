#include "torch/csrc/jit/api/module.h"

#include "core/compiler.h"
#include "core/util/prelude.h"
#include "trtorch/trtorch.h"
#include "NvInfer.h"

namespace trtorch {
CompileSpec::DataType::DataType(c10::ScalarType t) {
  TRTORCH_CHECK(t == at::kHalf || t == at::kFloat || t == at::kChar, "Data type is unsupported");
  switch (t) {
    case at::kHalf:
      value = DataType::kHalf;
      break;
    case at::kFloat:
    default:
      value = DataType::kFloat;
      break;
    case at::kChar:
      value = DataType::kChar;
  }
}

CompileSpec::Device::DeviceType::DeviceType(c10::DeviceType t) {
  TRTORCH_CHECK(t == at::kCUDA, "Device type when specified using torch device enum must be torch::kCUDA");
  value = DeviceType::kGPU;
}

CompileSpec::InputRange::InputRange(std::vector<int64_t> opt) {
  this->opt = opt;
  this->min = opt;
  this->max = opt;
}

CompileSpec::InputRange::InputRange(c10::IntArrayRef opt) {
  this->opt = core::util::toVec(opt);
  this->min = core::util::toVec(opt);
  this->max = core::util::toVec(opt);
}

CompileSpec::InputRange::InputRange(std::vector<int64_t> min, std::vector<int64_t> opt, std::vector<int64_t> max) {
  this->opt = opt;
  this->min = min;
  this->max = max;
}

CompileSpec::InputRange::InputRange(c10::IntArrayRef min, c10::IntArrayRef opt, c10::IntArrayRef max) {
  this->opt = core::util::toVec(opt);
  this->min = core::util::toVec(min);
  this->max = core::util::toVec(max);
}

CompileSpec::CompileSpec(std::vector<c10::ArrayRef<int64_t>> fixed_sizes) {
  for (auto in : fixed_sizes) {
    input_ranges.push_back(InputRange(in));
  }
}

CompileSpec::CompileSpec(std::vector<std::vector<int64_t>> fixed_sizes) {
  for (auto in : fixed_sizes) {
    input_ranges.push_back(InputRange(in));
  }
}

core::ir::InputRange to_internal_input_range(CompileSpec::InputRange i) {
  return core::ir::InputRange(i.min, i.opt, i.max);
}

std::vector<core::ir::InputRange> to_vec_internal_input_ranges(std::vector<CompileSpec::InputRange> external) {
  std::vector<core::ir::InputRange> internal;
  for (auto range : external) {
    internal.push_back(to_internal_input_range(range));
  }
  return internal;
}

core::CompileSpec to_internal_compile_spec(CompileSpec external) {
  core::CompileSpec internal(to_vec_internal_input_ranges(external.input_ranges));

  switch (external.op_precision) {
    case CompileSpec::DataType::kChar:
      internal.convert_info.engine_settings.op_precision = nvinfer1::DataType::kINT8;
      break;
    case CompileSpec::DataType::kHalf:
      internal.convert_info.engine_settings.op_precision = nvinfer1::DataType::kHALF;
      break;
    case CompileSpec::DataType::kFloat:
    default:
      internal.convert_info.engine_settings.op_precision = nvinfer1::DataType::kFLOAT;
  }

  std::vector<nvinfer1::DataType> enabled_precisions;
  for (std::string precision : external.enabled_precisions){
    if (precision.compare("fp32") == 0){
      enabled_precisions.push_back(nvinfer1::DataType::kFLOAT);
    }else if(precision.compare("fp16") == 0){
      enabled_precisions.push_back(nvinfer1::DataType::kHALF);
    }else if(precision.compare("int8") == 0){
      enabled_precisions.push_back(nvinfer1::DataType::kINT8);
    }else{
      TRTORCH_THROW_ERROR("Invalid precision provided. The choices of enabled_precisions can only be among these values fp32, fp16 or int8.");
    }
  }

  internal.convert_info.engine_settings.enabled_precisions = enabled_precisions;
  internal.convert_info.engine_settings.disable_tf32 = external.disable_tf32;
  internal.convert_info.engine_settings.refit = external.refit;
  internal.convert_info.engine_settings.debug = external.debug;
  internal.convert_info.engine_settings.truncate_long_and_double = external.truncate_long_and_double;
  internal.convert_info.engine_settings.strict_types = external.strict_types;
  internal.convert_info.engine_settings.device.allow_gpu_fallback = external.device.allow_gpu_fallback;
  internal.convert_info.engine_settings.max_batch_size = external.max_batch_size;
  internal.partition_info.enabled = external.torch_fallback.enabled;
  internal.partition_info.min_block_size = external.torch_fallback.min_block_size;
  internal.partition_info.forced_fallback_operators = external.torch_fallback.forced_fallback_ops;

  switch (external.device.device_type) {
    case CompileSpec::Device::DeviceType::kDLA:
      internal.convert_info.engine_settings.device.device_type = nvinfer1::DeviceType::kDLA;
      break;
    case CompileSpec::Device::DeviceType::kGPU:
    default:
      internal.convert_info.engine_settings.device.device_type = nvinfer1::DeviceType::kGPU;
  }

  switch (external.capability) {
    case CompileSpec::EngineCapability::kSAFE_GPU:
      internal.convert_info.engine_settings.capability = nvinfer1::EngineCapability::kSAFE_GPU;
      break;
    case CompileSpec::EngineCapability::kSAFE_DLA:
      internal.convert_info.engine_settings.capability = nvinfer1::EngineCapability::kSAFE_DLA;
      break;
    case CompileSpec::EngineCapability::kDEFAULT:
    default:
      internal.convert_info.engine_settings.capability = nvinfer1::EngineCapability::kDEFAULT;
  }

  internal.convert_info.engine_settings.device.gpu_id = external.device.gpu_id;
  internal.convert_info.engine_settings.device.dla_core = external.device.dla_core;
  internal.convert_info.engine_settings.num_min_timing_iters = external.num_min_timing_iters;
  internal.convert_info.engine_settings.num_avg_timing_iters = external.num_avg_timing_iters;
  internal.convert_info.engine_settings.workspace_size = external.workspace_size;

  if (internal.convert_info.engine_settings.op_precision == nvinfer1::DataType::kINT8) {
    internal.convert_info.engine_settings.calibrator = external.ptq_calibrator;
  } else {
    internal.convert_info.engine_settings.calibrator = nullptr;
  }

  return internal;
}

} // namespace trtorch
