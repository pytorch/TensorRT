#include <algorithm>

#include "torch/csrc/jit/api/module.h"

#include "core/compiler.h"
#include "core/util/prelude.h"

#include "trtorch/trtorch.h"

namespace trtorch {

nvinfer1::DataType toTRTDataType(CompileSpec::DataType value) {
  switch (value) {
    case CompileSpec::DataType::kChar:
      return nvinfer1::DataType::kINT8;
    case CompileSpec::DataType::kHalf:
      return nvinfer1::DataType::kHALF;
    case CompileSpec::DataType::kInt32:
      return nvinfer1::DataType::kINT32;
    case CompileSpec::DataType::kBool:
      return nvinfer1::DataType::kBOOL;
    case CompileSpec::DataType::kFloat:
    default:
      return nvinfer1::DataType::kFLOAT;
  }
}

nvinfer1::TensorFormat toTRTTensorFormat(CompileSpec::TensorFormat value) {
  switch (value) {
    case CompileSpec::TensorFormat::kChannelsLast:
      return nvinfer1::TensorFormat::kHWC;
    case CompileSpec::TensorFormat::kContiguous:
    default:
      return nvinfer1::TensorFormat::kLINEAR;
  }
}

CompileSpec::DataType::DataType(c10::ScalarType t) {
  TRTORCH_CHECK(
      t == at::kHalf || t == at::kFloat || t == at::kChar || t == at::kInt || t == at::kBool,
      "Data type is unsupported");
  switch (t) {
    case at::kHalf:
      value = DataType::kHalf;
      break;
    case at::kChar:
      value = DataType::kChar;
      break;
    case at::kInt:
      value = DataType::kInt32;
      break;
    case at::kBool:
      value = DataType::kBool;
      break;
    case at::kFloat:
    default:
      value = DataType::kFloat;
      break;
  }
}

CompileSpec::TensorFormat::TensorFormat(at::MemoryFormat t) {
  TRTORCH_CHECK(
    t == at::MemoryFormat::Contiguous || t == at::MemoryFormat::ChannelsLast, "Tensor format is unsupported"
  );

  switch (t) {
    case at::MemoryFormat::ChannelsLast:
      value = TensorFormat::kChannelsLast;
    case at::MemoryFormat::Contiguous:
    default:
      value = TensorFormat::kContiguous;
      break;
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

/* ====== DEFINE INPUTS CLASS MEMBERS ======*/
CompileSpec::Input::Input(std::vector<int64_t> shape, DataType dtype, TensorFormat format) {
  this->opt_shape = shape;
  this->min_shape = shape;
  this->max_shape = shape;
  this->shape = shape;
  this->dtype = dtype;
  this->format = format;
  this->input_is_dynamic = false;
}

CompileSpec::Input::Input(c10::IntArrayRef shape, DataType dtype, TensorFormat format) {
  this->opt_shape = core::util::toVec(shape);
  this->min_shape = core::util::toVec(shape);
  this->max_shape = core::util::toVec(shape);
  this->shape = core::util::toVec(shape);
  this->dtype = dtype;
  this->format = format;
  this->input_is_dynamic = false;
}

CompileSpec::Input::Input(std::vector<int64_t> min_shape, std::vector<int64_t> opt_shape, std::vector<int64_t> max_shape, DataType dtype, TensorFormat format) {
  this->opt_shape = opt_shape;
  this->min_shape = min_shape;
  this->max_shape = max_shape;
  this->shape = core::util::toVec(core::ir::Input(this->min_shape, this->opt_shape, this->max_shape).input_shape);
  this->dtype = dtype;
  this->format = format;
  this->input_is_dynamic = true;
}

CompileSpec::Input::Input(c10::IntArrayRef min_shape, c10::IntArrayRef opt_shape, c10::IntArrayRef max_shape, DataType dtype, TensorFormat format) {
  this->opt_shape = core::util::toVec(opt_shape);
  this->min_shape = core::util::toVec(min_shape);
  this->max_shape = core::util::toVec(max_shape);
  this->shape = core::util::toVec(core::ir::Input(this->min_shape, this->opt_shape, this->max_shape).input_shape);
  this->dtype = dtype;
  this->format = format;
  this->input_is_dynamic = true;
}

/* ==========================================*/

core::ir::Input to_internal_input(CompileSpec::InputRange& i) {
  return core::ir::Input(i.min, i.opt, i.max);
}

core::ir::Input to_internal_input(CompileSpec::Input& i) {
  return core::ir::Input(i.min_shape, i.opt_shape, i.max_shape, toTRTDataType(i.dtype), toTRTTensorFormat(i.format));
}

std::vector<core::ir::Input> to_vec_internal_inputs(std::vector<CompileSpec::InputRange>& external) {
  std::vector<core::ir::Input> internal;
  for (auto range : external) {
    internal.push_back(to_internal_input(range));
  }
  return internal;
}

std::vector<core::ir::Input> to_vec_internal_inputs(std::vector<CompileSpec::Input>& external) {
  std::vector<core::ir::Input> internal;
  for (auto range : external) {
    internal.push_back(to_internal_input(range));
  }
  return internal;
}

core::CompileSpec to_internal_compile_spec(CompileSpec external) {
  core::CompileSpec internal(to_vec_internal_inputs(external.inputs));
  if (external.input_ranges.size() > 0 ) {
    internal = core::CompileSpec(to_vec_internal_inputs(external.input_ranges));
  } else {
    TRTORCH_CHECK(external.inputs.size() > 0, "Compilation requires at least one input specification");
    internal = core::CompileSpec(to_vec_internal_inputs(external.inputs));
  }

  if (external.enabled_precisions.size() <= 1 && toTRTDataType(external.op_precision) != nvinfer1::DataType::kFLOAT) {
    internal.convert_info.engine_settings.enabled_precisions.insert(toTRTDataType(external.op_precision));
  } else {
    for(auto p : external.enabled_precisions) {
      internal.convert_info.engine_settings.enabled_precisions.insert(toTRTDataType(p));
    }
  }

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

  if (internal.convert_info.engine_settings.enabled_precisions.find(nvinfer1::DataType::kINT8) != internal.convert_info.engine_settings.enabled_precisions.end()) {
    internal.convert_info.engine_settings.calibrator = external.ptq_calibrator;
  } else {
    internal.convert_info.engine_settings.calibrator = nullptr;
  }

  return internal;
}

} // namespace trtorch
