
#include "tensorrt_classes.h"

namespace trtorch {
namespace pyapi {

std::string InputRange::to_str() {
  auto vec_to_str = [](std::vector<int64_t> shape) -> std::string {
    std::stringstream ss;
    ss << '[';
    for (auto i : shape) {
      ss << i << ',';
    }
    ss << ']';
    return ss.str();
  };

  std::stringstream ss;
  ss << "        {" << std::endl;
  ss << "            min: " << vec_to_str(min) << ',' << std::endl;
  ss << "            opt: " << vec_to_str(opt) << ',' << std::endl;
  ss << "            max: " << vec_to_str(max) << ',' << std::endl;
  ss << "        }" << std::endl;
  return ss.str();
}

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
    default:
      return "Float";
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
    case DataType::kBool:
      return nvinfer1::DataType::kBOOL;
    case DataType::kFloat:
    default:
      return nvinfer1::DataType::kFLOAT;
  }
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
      return nvinfer1::EngineCapability::kSAFE_DLA;
    case EngineCapability::kSAFE_GPU:
      return nvinfer1::EngineCapability::kSAFE_GPU;
    case EngineCapability::kDEFAULT:
    default:
      return nvinfer1::EngineCapability::kDEFAULT;
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
  ss << "    }" << std::endl;
  return ss.str();
}

core::CompileSpec CompileSpec::toInternalCompileSpec() {
  std::vector<core::ir::InputRange> internal_input_ranges;
  for (auto i : input_ranges) {
    internal_input_ranges.push_back(i.toInternalInputRange());
  }

  std::vector<nvinfer1::DataType> trt_input_dtypes;
  for (auto dtype : input_dtypes) {
    trt_input_dtypes.push_back(toTRTDataType(dtype));
  }

  auto info = core::CompileSpec(internal_input_ranges);
  info.convert_info.engine_settings.op_precision = toTRTDataType(op_precision);
  info.convert_info.engine_settings.input_dtypes = trt_input_dtypes;
  info.convert_info.engine_settings.calibrator = ptq_calibrator;
  info.convert_info.engine_settings.disable_tf32 = disable_tf32;
  info.convert_info.engine_settings.refit = refit;
  info.convert_info.engine_settings.debug = debug;
  info.convert_info.engine_settings.strict_types = strict_types;
  info.convert_info.engine_settings.device.device_type = toTRTDeviceType(device.device_type);
  info.convert_info.engine_settings.device.gpu_id = device.gpu_id;
  info.convert_info.engine_settings.device.dla_core = device.dla_core;
  info.convert_info.engine_settings.device.allow_gpu_fallback = device.allow_gpu_fallback;
  info.partition_info.enabled = torch_fallback.enabled;
  info.partition_info.min_block_size = torch_fallback.min_block_size;
  info.partition_info.forced_fallback_operators = torch_fallback.forced_fallback_operators;
  info.convert_info.engine_settings.truncate_long_and_double = truncate_long_and_double;

  info.convert_info.engine_settings.capability = toTRTEngineCapability(capability);
  TRTORCH_CHECK(num_min_timing_iters >= 0, "num_min_timing_iters must be 0 or greater");
  info.convert_info.engine_settings.num_min_timing_iters = num_min_timing_iters;
  TRTORCH_CHECK(num_avg_timing_iters >= 0, "num_avg_timing_iters must be 0 or greater");
  info.convert_info.engine_settings.num_avg_timing_iters = num_avg_timing_iters;
  TRTORCH_CHECK(workspace_size >= 0, "workspace_size must be 0 or greater");
  info.convert_info.engine_settings.workspace_size = workspace_size;
  TRTORCH_CHECK(max_batch_size >= 0, "max_batch_size must be 0 or greater");
  info.convert_info.engine_settings.max_batch_size = max_batch_size;
  return info;
}

std::string CompileSpec::stringify() {
  std::stringstream ss;
  ss << "TensorRT Compile Spec: {" << std::endl;
  ss << "    \"Input Shapes\": [" << std::endl;
  for (auto i : input_ranges) {
    ss << i.to_str();
  }
  ss << "    ]" << std::endl;
  ss << "    \"Op Precision\": " << to_str(op_precision) << std::endl;
  ss << "    \"Input dtypes\": [" << std::endl;
  for (auto i : input_dtypes) {
    ss << to_str(i);
  }
  ss << "    ]" << std::endl;
  ss << "    \"TF32 Disabled\": " << disable_tf32 << std::endl;
  ss << "    \"Refit\": " << refit << std::endl;
  ss << "    \"Debug\": " << debug << std::endl;
  ss << "    \"Strict Types\": " << strict_types << std::endl;
  ss << "    \"Device\": " << device.to_str() << std::endl;
  ss << "    \"Engine Capability\": " << to_str(capability) << std::endl;
  ss << "    \"Num Min Timing Iters\": " << num_min_timing_iters << std::endl;
  ss << "    \"Num Avg Timing Iters\": " << num_avg_timing_iters << std::endl;
  ss << "    \"Workspace Size\": " << workspace_size << std::endl;
  ss << "    \"Max Batch Size\": " << max_batch_size << std::endl;
  ss << "    \"Truncate long and double\": " << truncate_long_and_double << std::endl;
  ss << "    \"Torch Fallback\": " << torch_fallback.to_str();
  ss << "}";
  return ss.str();
}

} // namespace pyapi
} // namespace trtorch
