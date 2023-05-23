
#include "tensorrt_classes.h"

namespace torch_tensorrt {
namespace pyapi {

std::string to_str(DataType value) {
  switch (value) {
    case DataType::kHalf:
      return "Half";
    case DataType::kChar:
      return "Int8";
    case DataType::kInt32:
      return "Int32";
    case DataType::kBool:
      return "Bool";
    case DataType::kFloat:
      return "Float";
    case DataType::kLong:
      return "Long";
    default:
      return "Unknown data type";
  }
}

nvinfer1::DataType toTRTDataType(DataType value) {
  switch (value) {
    case DataType::kChar:
      return nvinfer1::DataType::kINT8;
    case DataType::kHalf:
      return nvinfer1::DataType::kHALF;
    case DataType::kInt32:
      return nvinfer1::DataType::kINT32;
    case DataType::kLong:
      return nvinfer1::DataType::kINT32;
    case DataType::kBool:
      return nvinfer1::DataType::kBOOL;
    case DataType::kFloat:
      return nvinfer1::DataType::kFLOAT;
    case DataType::kUnknown:
      return nvinfer1::DataType::kFLOAT;
    default:
      TORCHTRT_THROW_ERROR("Unknown data type: " << to_str(value));
  }
}

at::ScalarType toAtenDataType(DataType value) {
  switch (value) {
    case DataType::kChar:
      return at::kChar;
    case DataType::kHalf:
      return at::kHalf;
    case DataType::kInt32:
      return at::kInt;
    case DataType::kLong:
      return at::kLong;
    case DataType::kBool:
      return at::kBool;
    case DataType::kFloat:
      return at::kFloat;
    case DataType::kUnknown:
      return at::kFloat;
    default:
      TORCHTRT_THROW_ERROR("Unknown data type: " << to_str(value));
  }
}

Device::Device(const core::runtime::RTDevice& internal_dev) {
  device_type = DeviceType::kGPU;
  gpu_id = internal_dev.id;
  dla_core = -1;
  allow_gpu_fallback = false;
}

nvinfer1::TensorFormat toTRTTensorFormat(TensorFormat value) {
  switch (value) {
    case TensorFormat::kChannelsLast:
      return nvinfer1::TensorFormat::kHWC;
    case TensorFormat::kContiguous:
    default:
      return nvinfer1::TensorFormat::kLINEAR;
  }
}

std::string to_str(TensorFormat value) {
  switch (value) {
    case TensorFormat::kContiguous:
      return "Contiguous/Linear/NCHW";
    case TensorFormat::kChannelsLast:
      return "Channel Last/NHWC";
    default:
      return "UNKNOWN";
  }
}

core::ir::Input Input::toInternalInput() {
  if (!input_is_dynamic) {
    return core::ir::Input(opt, toAtenDataType(dtype), toTRTTensorFormat(format), explicit_set_dtype, tensor_domain);
  } else {
    return core::ir::Input(
        min, opt, max, toAtenDataType(dtype), toTRTTensorFormat(format), explicit_set_dtype, tensor_domain);
  }
}

std::string Input::to_str() {
  auto vec_to_str = [](std::vector<int64_t> shape) -> std::string {
    std::stringstream ss;
    ss << '(';
    for (auto i : shape) {
      ss << i << ',';
    }
    ss << ')';
    return ss.str();
  };

  auto domain_to_str = [](std::vector<double> domain) -> std::string {
    std::stringstream ss;
    ss << "[" << domain[0] << ", " << domain[1] << ")";
    return ss.str();
  };

  std::stringstream ss;
  ss << "Input(";

  if (!input_is_dynamic) {
    ss << "shape=" << vec_to_str(opt) << ", ";
  } else {
    ss << "min_shape=" << vec_to_str(min) << ", ";
    ss << "opt_shape=" << vec_to_str(opt) << ", ";
    ss << "max_shape=" << vec_to_str(max) << ", ";
  }

  ss << "dtype=" << pyapi::to_str(dtype) << ", ";
  ss << "format=" << pyapi::to_str(format) << ", ";
  ss << "tensor_domain=" << domain_to_str(tensor_domain) << ")";

  return ss.str();
}

std::string sig_to_str(torch::jit::IValue input_sig) {
  if (input_sig.isTuple()) {
    auto input_tuple = input_sig.toTuple();
    std::vector<std::string> children;
    for (auto item : input_tuple->elements()) {
      auto child = sig_to_str(item);
      children.push_back(child);
    }
    std::stringstream ss;
    ss << "(";
    for (auto i : children) {
      ss << i << ", ";
    }
    ss << ")";
    return ss.str();
  } else if (input_sig.isList()) {
    auto input_list = input_sig.toList().vec();
    std::vector<std::string> children;
    for (auto item : input_list) {
      auto child = sig_to_str(item);
      children.push_back(child);
    }
    std::stringstream ss;
    ss << "[";
    for (auto i : children) {
      ss << i << ", ";
    }
    ss << "]";
    return ss.str();
  } else if (input_sig.isCustomClass()) {
    auto cur_input = input_sig.toCustomClass<Input>();
    return cur_input->to_str();
  } else if (input_sig.isPyObject()) {
    auto py_object_holder = input_sig.toPyObjectHolder();
    auto infer_type = py_object_holder->tryToInferType();
    auto type = infer_type.type();
    torch::jit::IValue ival = py_object_holder->toIValue(type);
    torch::jit::IValue converted_item;
    return sig_to_str(ival);
  } else {
    LOG_ERROR("Unknown input spec type");
    return "";
  }
}

std::string InputSignature::to_str() {
  std::stringstream ss;
  return sig_to_str(signature_ivalue);
}

std::string to_str(DeviceType value) {
  switch (value) {
    case DeviceType::kDLA:
      return "DLA";
    case DeviceType::kGPU:
    default:
      return "GPU";
  }
}

nvinfer1::DeviceType toTRTDeviceType(DeviceType value) {
  switch (value) {
    case DeviceType::kDLA:
      return nvinfer1::DeviceType::kDLA;
    case DeviceType::kGPU:
    default:
      return nvinfer1::DeviceType::kGPU;
  }
}

core::runtime::RTDevice Device::toInternalRTDevice() {
  return core::runtime::RTDevice(gpu_id, toTRTDeviceType(device_type));
}

std::string Device::toSerializedRTDevice() {
  return this->toInternalRTDevice().serialize();
}

std::string Device::to_str() {
  std::stringstream ss;
  std::string fallback = allow_gpu_fallback ? "True" : "False";
  ss << " {" << std::endl;
  ss << "        \"device_type\": " << pyapi::to_str(device_type) << std::endl;
  ss << "        \"allow_gpu_fallback\": " << fallback << std::endl;
  ss << "        \"gpu_id\": " << gpu_id << std::endl;
  ss << "        \"dla_core\": " << dla_core << std::endl;
  ss << "    }" << std::endl;
  return ss.str();
}

std::string to_str(EngineCapability value) {
  switch (value) {
    case EngineCapability::kSAFE_GPU:
      return "Safe GPU";
    case EngineCapability::kSAFE_DLA:
      return "Safe DLA";
    case EngineCapability::kDEFAULT:
    default:
      return "Default";
  }
}

nvinfer1::EngineCapability toTRTEngineCapability(EngineCapability value) {
  switch (value) {
    case EngineCapability::kSAFE_DLA:
      return TRT_ENGINE_CAPABILITY_DLA_STANDALONE;
    case EngineCapability::kSAFE_GPU:
      return TRT_ENGINE_CAPABILITY_SAFETY;
    case EngineCapability::kDEFAULT:
    default:
      return TRT_ENGINE_CAPABILITY_STANDARD;
  }
}

std::string TorchFallback::to_str() {
  std::stringstream ss;
  std::string e = enabled ? "True" : "False";
  ss << " {" << std::endl;
  ss << "        \"enabled\": " << e << std::endl;
  ss << "        \"min_block_size\": " << min_block_size << std::endl;
  ss << "        \"forced_fallback_operators\": [" << std::endl;
  for (auto i : forced_fallback_operators) {
    ss << "            " << i << ',' << std::endl;
  }
  ss << "        ]" << std::endl;
  ss << "        \"forced_fallback_modules\": [" << std::endl;
  for (auto i : forced_fallback_modules) {
    ss << "            " << i << ',' << std::endl;
  }
  ss << "        ]" << std::endl;
  ss << "    }" << std::endl;
  return ss.str();
}

void to_internal_input_signature(torch::jit::IValue input_ivalue, torch::jit::IValue& converted_ivalue) {
  if (input_ivalue.isTuple()) {
    auto input_tuple = input_ivalue.toTuple();
    std::vector<torch::jit::IValue> converted_elements;
    for (auto item : input_tuple->elements()) {
      torch::jit::IValue converted_item;
      to_internal_input_signature(item, converted_item);
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
      to_internal_input_signature(item, converted_item);
      converted_elements.push_back(converted_item);
    }
    converted_ivalue = torch::jit::IValue(converted_elements);
  } else if (input_ivalue.isCustomClass()) {
    core::ir::Input cur_input = (*(input_ivalue.toCustomClass<Input>())).toInternalInput();
    converted_ivalue = torch::jit::IValue(std::move(c10::make_intrusive<core::ir::Input>(cur_input)));
  } else if (input_ivalue.isPyObject()) {
    auto py_object_holder = input_ivalue.toPyObjectHolder();
    auto infer_type = py_object_holder->tryToInferType();
    auto type = infer_type.type();
    torch::jit::IValue ival = py_object_holder->toIValue(type);
    torch::jit::IValue converted_item;
    to_internal_input_signature(ival, converted_item);
    converted_ivalue = torch::jit::IValue(converted_item);
  } else {
    LOG_ERROR("Unknown input spec type");
  }
}

core::CompileSpec init_compile_spec(CompileSpec external) {
  if (external.inputs.size() > 0) {
    LOG_DEBUG("init_compile_spec with input vector");
    std::vector<core::ir::Input> internal_inputs;
    for (auto i : external.inputs) {
      internal_inputs.push_back(i.toInternalInput());
    }
    core::CompileSpec internal(internal_inputs);
    return internal;
  } else {
    LOG_DEBUG("init_compile_spec with input signature");
    torch::jit::IValue converted_input_signature;
    to_internal_input_signature(external.input_signature.signature_ivalue, converted_input_signature);
    core::CompileSpec internal(converted_input_signature);
    return internal;
  }
}

core::CompileSpec CompileSpec::toInternalCompileSpec(bool converting_to_trt_engine) {
  core::CompileSpec info = init_compile_spec(*this);

  info.lower_info.converting_to_trt_engine = converting_to_trt_engine;

  for (auto p : enabled_precisions) {
    info.convert_info.engine_settings.enabled_precisions.insert(toTRTDataType(p));
  }

  info.partitioning_info.cast_int8_inputs = true;

  if (ptq_calibrator) {
    info.convert_info.engine_settings.calibrator = ptq_calibrator;
    info.partitioning_info.cast_int8_inputs = false;
  } else {
    if (info.convert_info.engine_settings.enabled_precisions.find(nvinfer1::DataType::kINT8) !=
        info.convert_info.engine_settings.enabled_precisions.end()) {
      info.partitioning_info.cast_int8_inputs = false;
      info.lower_info.unfreeze_module = true;
      info.lower_info.disable_cse = true;
    }
  }
  info.convert_info.engine_settings.sparse_weights = sparse_weights;
  info.convert_info.engine_settings.disable_tf32 = disable_tf32;
  info.convert_info.engine_settings.refit = refit;
  info.convert_info.engine_settings.debug = debug;

  // Specify + replicate device settings for phases requiring it
  info.convert_info.engine_settings.device.device_type = toTRTDeviceType(device.device_type);
  info.convert_info.engine_settings.device.gpu_id = device.gpu_id;
  info.convert_info.engine_settings.device.dla_core = device.dla_core;
  info.convert_info.engine_settings.device.allow_gpu_fallback = device.allow_gpu_fallback;

  info.lower_info.target_device.device_type = toTRTDeviceType(device.device_type);
  info.lower_info.target_device.gpu_id = device.gpu_id;
  info.lower_info.target_device.dla_core = device.dla_core;
  info.lower_info.target_device.allow_gpu_fallback = device.allow_gpu_fallback;

  info.partitioning_info.target_device.device_type = toTRTDeviceType(device.device_type);
  info.partitioning_info.target_device.gpu_id = device.gpu_id;
  info.partitioning_info.target_device.dla_core = device.dla_core;
  info.partitioning_info.target_device.allow_gpu_fallback = device.allow_gpu_fallback;

  info.partitioning_info.enabled = torch_fallback.enabled;
  info.partitioning_info.min_block_size = torch_fallback.min_block_size;
  info.partitioning_info.forced_fallback_operators = torch_fallback.forced_fallback_operators;
  info.partitioning_info.truncate_long_and_double = truncate_long_and_double;
  info.lower_info.forced_fallback_modules = torch_fallback.forced_fallback_modules;
  info.convert_info.engine_settings.truncate_long_and_double = truncate_long_and_double;
  info.convert_info.engine_settings.allow_shape_tensors = allow_shape_tensors;

  info.convert_info.engine_settings.capability = toTRTEngineCapability(capability);
  TORCHTRT_CHECK(num_avg_timing_iters >= 0, "num_avg_timing_iters must be 0 or greater");
  info.convert_info.engine_settings.num_avg_timing_iters = num_avg_timing_iters;
  TORCHTRT_CHECK(workspace_size >= 0, "workspace_size must be 0 or greater");
  info.convert_info.engine_settings.workspace_size = workspace_size;
  TORCHTRT_CHECK(
      dla_sram_size >= 4096,
      "DLA managed SRAM size must be at least 4 KiB and must be a power of 2. This defaults to 1 MiB");
  info.convert_info.engine_settings.dla_sram_size = dla_sram_size;
  TORCHTRT_CHECK(
      dla_local_dram_size >= 4096,
      "DLA Local DRAM size must be at least 4 KiB and must be a power of 2. This defaults to 1 GiB");
  info.convert_info.engine_settings.dla_local_dram_size = dla_local_dram_size;
  TORCHTRT_CHECK(
      dla_global_dram_size >= 4096,
      "DLA Global DRAM size must be at least 4 KiB and must be a power of 2. This defaults to 512 MiB");
  info.convert_info.engine_settings.dla_global_dram_size = dla_global_dram_size;
  return info;
}

std::string CompileSpec::stringify() {
  std::stringstream ss;
  ss << "TensorRT Compile Spec: {" << std::endl;
  if (inputs.size() > 0) {
    ss << "    \"Inputs\": [" << std::endl;
    for (auto i : inputs) {
      ss << i.to_str();
    }
    ss << "    ]" << std::endl;
  } else {
    ss << "    \"Input Signature\": " << input_signature.to_str() << std::endl;
  }
  ss << "    \"Enabled Precision\": [";
  for (auto p : enabled_precisions) {
    ss << to_str(p) << ", ";
  }
  ss << "]" << std::endl;
  ss << "    \"TF32 Disabled\": " << disable_tf32 << std::endl;
  ss << "    \"Sparsity\": " << sparse_weights << std::endl;
  ss << "    \"Refit\": " << refit << std::endl;
  ss << "    \"Debug\": " << debug << std::endl;
  ss << "    \"Device\": " << device.to_str() << std::endl;
  ss << "    \"Engine Capability\": " << to_str(capability) << std::endl;
  ss << "    \"Num Avg Timing Iters\": " << num_avg_timing_iters << std::endl;
  ss << "    \"Workspace Size\": " << workspace_size << std::endl;
  ss << "    \"DLA SRAM Size\": " << dla_sram_size << std::endl;
  ss << "    \"DLA Local DRAM Size\": " << dla_local_dram_size << std::endl;
  ss << "    \"DLA Global DRAM Size\": " << dla_global_dram_size << std::endl;
  ss << "    \"Truncate long and double\": " << truncate_long_and_double << std::endl;
  ss << "    \"Allow Shape tensors\": " << allow_shape_tensors << std::endl;
  ss << "    \"Torch Fallback\": " << torch_fallback.to_str();
  ss << "}";
  return ss.str();
}

} // namespace pyapi
} // namespace torch_tensorrt
