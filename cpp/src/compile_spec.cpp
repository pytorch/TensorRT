#include <algorithm>

#include "torch/csrc/jit/api/module.h"

#include "core/compiler.h"
#include "core/util/prelude.h"

#include "torch_tensorrt/torch_tensorrt.h"

namespace torch_tensorrt {
// Defined in types.cpp
nvinfer1::DataType toTRTDataType(DataType value);
nvinfer1::TensorFormat toTRTTensorFormat(TensorFormat value);
torchtrt::core::ir::Input to_internal_input(Input& i);
std::vector<torchtrt::core::ir::Input> to_vec_internal_inputs(std::vector<Input>& external);
torchtrt::core::runtime::CudaDevice to_internal_cuda_device(Device device);

namespace torchscript {
CompileSpec::CompileSpec(std::vector<c10::ArrayRef<int64_t>> fixed_sizes) {
  for (auto in : fixed_sizes) {
    inputs.push_back(Input(in));
  }
}

CompileSpec::CompileSpec(std::vector<std::vector<int64_t>> fixed_sizes) {
  for (auto in : fixed_sizes) {
    inputs.push_back(Input(in));
  }
}

torchtrt::core::CompileSpec to_internal_compile_spec(CompileSpec external) {
  torchtrt::core::CompileSpec internal(to_vec_internal_inputs(external.inputs));

  for (auto p : external.enabled_precisions) {
    internal.convert_info.engine_settings.enabled_precisions.insert(toTRTDataType(p));
  }

  internal.convert_info.engine_settings.sparse_weights = external.sparse_weights;
  internal.convert_info.engine_settings.disable_tf32 = external.disable_tf32;
  internal.convert_info.engine_settings.refit = external.refit;
  internal.convert_info.engine_settings.debug = external.debug;
  internal.convert_info.engine_settings.truncate_long_and_double = external.truncate_long_and_double;
  internal.convert_info.engine_settings.strict_types = external.strict_types;
  internal.convert_info.engine_settings.device.allow_gpu_fallback = external.device.allow_gpu_fallback;
  internal.convert_info.engine_settings.max_batch_size = external.max_batch_size;

  TORCHTRT_CHECK(
      !(external.require_full_compilation && (external.torch_executed_ops.size() > 0)),
      "require_full_compilation is enabled however the list of ops to run in torch is not empty (Found "
          << external.torch_executed_ops.size() << " ops)");

  TORCHTRT_CHECK(
      !(external.require_full_compilation && (external.torch_executed_modules.size() > 0)),
      "require_full_compilation is enabled however the list of modules to run in torch is not empty (Found "
          << external.torch_executed_modules.size() << " modules)");

  internal.partition_info.enabled = !external.require_full_compilation;
  internal.partition_info.min_block_size = external.min_block_size;
  internal.partition_info.forced_fallback_operators = std::move(external.torch_executed_ops);
  internal.lower_info.forced_fallback_modules = std::move(external.torch_executed_modules);

  switch (external.device.device_type) {
    case Device::DeviceType::kDLA:
      internal.convert_info.engine_settings.device.device_type = nvinfer1::DeviceType::kDLA;
      break;
    case Device::DeviceType::kGPU:
    default:
      internal.convert_info.engine_settings.device.device_type = nvinfer1::DeviceType::kGPU;
  }

  switch (external.capability) {
    case EngineCapability::kSAFETY:
      internal.convert_info.engine_settings.capability = TRT_ENGINE_CAPABILITY_SAFETY;
      break;
    case EngineCapability::kDLA_STANDALONE:
      internal.convert_info.engine_settings.capability = TRT_ENGINE_CAPABILITY_DLA_STANDALONE;
      break;
    case EngineCapability::kSTANDARD:
    default:
      internal.convert_info.engine_settings.capability = TRT_ENGINE_CAPABILITY_STANDARD;
  }

  internal.convert_info.engine_settings.device.gpu_id = external.device.gpu_id;
  internal.convert_info.engine_settings.device.dla_core = external.device.dla_core;
  internal.convert_info.engine_settings.num_min_timing_iters = external.num_min_timing_iters;
  internal.convert_info.engine_settings.num_avg_timing_iters = external.num_avg_timing_iters;
  internal.convert_info.engine_settings.workspace_size = external.workspace_size;

  if (internal.convert_info.engine_settings.enabled_precisions.find(nvinfer1::DataType::kINT8) !=
      internal.convert_info.engine_settings.enabled_precisions.end()) {
    if (external.ptq_calibrator) {
      internal.convert_info.engine_settings.calibrator = external.ptq_calibrator;
    } else {
      internal.lower_info.unfreeze_module = true;
      internal.lower_info.disable_cse = true;
      internal.convert_info.engine_settings.calibrator = nullptr;
    }
  } else {
    internal.convert_info.engine_settings.calibrator = nullptr;
  }

  return internal;
}
} // namespace torchscript
} // namespace torch_tensorrt
