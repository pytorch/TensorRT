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
torchtrt::core::runtime::RTDevice to_internal_rt_device(Device device);

namespace torchscript {
CompileSpec::CompileSpec(std::vector<c10::ArrayRef<int64_t>> fixed_sizes) {
  for (auto in : fixed_sizes) {
    graph_inputs.inputs.push_back(Input(in));
  }
}

CompileSpec::CompileSpec(std::vector<std::vector<int64_t>> fixed_sizes) {
  for (auto in : fixed_sizes) {
    graph_inputs.inputs.push_back(Input(in));
  }
}

CompileSpec::CompileSpec(std::vector<Input> inputs) {
  graph_inputs.inputs = std::move(inputs);
}

CompileSpec::CompileSpec(torch::jit::IValue input_signature) {
  graph_inputs.input_signature = input_signature;
}

void to_internal_input_signature(torch::jit::IValue input_ivalue, torch::jit::IValue& converted_ivalue, int depth = 0) {
  TORCHTRT_CHECK(
      depth <= 2, "Input nesting depth exceeds max supported depth, use 1 level: [A, B], or 2 level: [A, (B, C)]")
  if (input_ivalue.isTuple()) {
    auto input_tuple = input_ivalue.toTuple();
    std::vector<torch::jit::IValue> converted_elements;
    for (auto item : input_tuple->elements()) {
      torch::jit::IValue converted_item;
      to_internal_input_signature(item, converted_item, depth++);
      converted_elements.push_back(converted_item);
      auto tuple_ptr = c10::ivalue::Tuple::create(converted_elements);
      converted_ivalue = torch::jit::IValue(tuple_ptr);
    }
  } else if (input_ivalue.isList()) {
    auto input_list = input_ivalue.toList().vec();
    c10::TypePtr type = input_list[0].type();
    auto converted_elements = c10::impl::GenericList(type);
    for (auto item : input_list) {
      torch::jit::IValue converted_item;
      to_internal_input_signature(item, converted_item, depth++);
      converted_elements.push_back(converted_item);
    }
    converted_ivalue = torch::jit::IValue(converted_elements);
  } else if (input_ivalue.isCustomClass()) {
    torchtrt::core::ir::Input cur_input = to_internal_input(*(input_ivalue.toCustomClass<torchtrt::Input>()));
    converted_ivalue = torch::jit::IValue(std::move(c10::make_intrusive<torch_tensorrt::core::ir::Input>(cur_input)));
  }
}

torchtrt::core::CompileSpec init_compile_spec(CompileSpec& external) {
  if (external.graph_inputs.inputs.size() > 0) {
    torchtrt::core::CompileSpec internal(to_vec_internal_inputs(external.graph_inputs.inputs));
    return internal;
  } else {
    torch::jit::IValue converted_input_signature;
    LOG_WARNING("Input signature parsing is an experimental feature, behavior and APIs may change");
    to_internal_input_signature(external.graph_inputs.input_signature, converted_input_signature);
    torchtrt::core::CompileSpec internal(converted_input_signature);
    return internal;
  }
}

torchtrt::core::CompileSpec to_internal_compile_spec(CompileSpec external, bool converting_to_trt_engine) {
  torchtrt::core::CompileSpec internal = init_compile_spec(external);

  internal.lower_info.converting_to_trt_engine = converting_to_trt_engine;

  for (auto p : external.enabled_precisions) {
    internal.convert_info.engine_settings.enabled_precisions.insert(toTRTDataType(p));
  }

  internal.convert_info.engine_settings.sparse_weights = external.sparse_weights;
  internal.convert_info.engine_settings.disable_tf32 = external.disable_tf32;
  internal.convert_info.engine_settings.refit = external.refit;
  internal.convert_info.engine_settings.debug = external.debug;
  internal.convert_info.engine_settings.truncate_long_and_double = external.truncate_long_and_double;
  internal.convert_info.engine_settings.allow_shape_tensors = external.allow_shape_tensors;
  internal.convert_info.engine_settings.device.allow_gpu_fallback = external.device.allow_gpu_fallback;
  internal.lower_info.target_device.allow_gpu_fallback = external.device.allow_gpu_fallback;
  internal.partitioning_info.target_device.allow_gpu_fallback = external.device.allow_gpu_fallback;

  TORCHTRT_CHECK(
      !(external.require_full_compilation && (external.torch_executed_ops.size() > 0)),
      "require_full_compilation is enabled however the list of ops to run in torch is not empty (Found "
          << external.torch_executed_ops.size() << " ops)");

  TORCHTRT_CHECK(
      !(external.require_full_compilation && (external.torch_executed_modules.size() > 0)),
      "require_full_compilation is enabled however the list of modules to run in torch is not empty (Found "
          << external.torch_executed_modules.size() << " modules)");

  internal.partitioning_info.enabled = !external.require_full_compilation;
  internal.partitioning_info.min_block_size = external.min_block_size;
  internal.partitioning_info.forced_fallback_operators = std::move(external.torch_executed_ops);
  internal.partitioning_info.truncate_long_and_double = external.truncate_long_and_double;
  internal.lower_info.forced_fallback_modules = std::move(external.torch_executed_modules);

  switch (external.device.device_type) {
    case Device::DeviceType::kDLA:
      internal.convert_info.engine_settings.device.device_type = nvinfer1::DeviceType::kDLA;
      internal.lower_info.target_device.device_type = nvinfer1::DeviceType::kDLA;
      internal.partitioning_info.target_device.device_type = nvinfer1::DeviceType::kDLA;
      break;
    case Device::DeviceType::kGPU:
    default:
      internal.convert_info.engine_settings.device.device_type = nvinfer1::DeviceType::kGPU;
      internal.lower_info.target_device.device_type = nvinfer1::DeviceType::kGPU;
      internal.partitioning_info.target_device.device_type = nvinfer1::DeviceType::kGPU;
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
  internal.lower_info.target_device.gpu_id = external.device.gpu_id;
  internal.lower_info.target_device.dla_core = external.device.dla_core;
  internal.partitioning_info.target_device.gpu_id = external.device.gpu_id;
  internal.partitioning_info.target_device.dla_core = external.device.dla_core;

  internal.convert_info.engine_settings.num_avg_timing_iters = external.num_avg_timing_iters;
  internal.convert_info.engine_settings.workspace_size = external.workspace_size;
  internal.convert_info.engine_settings.dla_sram_size = external.dla_sram_size;
  internal.convert_info.engine_settings.dla_local_dram_size = external.dla_local_dram_size;
  internal.convert_info.engine_settings.dla_global_dram_size = external.dla_global_dram_size;

  internal.partitioning_info.cast_int8_inputs = true;

  if (internal.convert_info.engine_settings.enabled_precisions.find(nvinfer1::DataType::kINT8) !=
      internal.convert_info.engine_settings.enabled_precisions.end()) {
    internal.partitioning_info.cast_int8_inputs = false;
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
