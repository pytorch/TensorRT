#include <algorithm>

#include "torch/csrc/jit/api/module.h"

#include "core/compiler.h"
#include "core/util/prelude.h"

#include "torch_tensorrt/torch_tensorrt.h"

namespace torch_tensorrt {
std::ostream& operator<<(std::ostream& os, const DataType& dtype) {
  switch (dtype) {
    case DataType::kChar:
      os << "char";
      break;
    case DataType::kHalf:
      os << "half";
      break;
    case DataType::kInt:
      os << "int";
      break;
    case DataType::kBool:
      os << "bool";
      break;
    case DataType::kFloat:
      os << "float";
      break;
    case DataType::kUnknown:
    default:
      os << "unknown";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const TensorFormat& format) {
  switch (format) {
    case TensorFormat::kChannelsLast:
      os << "channels last";
      break;
    case TensorFormat::kContiguous:
      os << "contiguous";
      break;
    case TensorFormat::kUnknown:
    default:
      os << "unknown";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const Input& input) {
  auto vec_to_str = [](std::vector<int64_t> shape) -> std::string {
    std::stringstream ss;
    ss << '[';
    for (auto i : shape) {
      ss << i << ',';
    }
    ss << ']';
    return ss.str();
  };

  if (!input.input_is_dynamic) {
    os << "Input(shape: " << vec_to_str(input.shape) << ", dtype: " << input.dtype << ", format: " << input.format
       << ')';
  } else {
    os << "Input(shape: " << vec_to_str(input.shape) << ", min: " << vec_to_str(input.min_shape)
       << ", opt: " << vec_to_str(input.opt_shape) << ", max: " << vec_to_str(input.max_shape)
       << ", dtype: " << input.dtype << ", format: " << input.format << ')';
  }
  return os;
}

nvinfer1::DataType toTRTDataType(DataType value) {
  switch (value) {
    case DataType::kChar:
      return nvinfer1::DataType::kINT8;
    case DataType::kHalf:
      return nvinfer1::DataType::kHALF;
    case DataType::kInt:
      return nvinfer1::DataType::kINT32;
    case DataType::kBool:
      return nvinfer1::DataType::kBOOL;
    case DataType::kFloat:
    default:
      return nvinfer1::DataType::kFLOAT;
  }
}

nvinfer1::TensorFormat toTRTTensorFormat(TensorFormat value) {
  TORCHTRT_CHECK(!(value == TensorFormat::kUnknown), "Tensor format is unknown");
  switch (value) {
    case TensorFormat::kChannelsLast:
      return nvinfer1::TensorFormat::kHWC;
    case TensorFormat::kContiguous:
    default:
      return nvinfer1::TensorFormat::kLINEAR;
  }
}

DataType::DataType(c10::ScalarType t) {
  TORCHTRT_CHECK(
      t == at::kHalf || t == at::kFloat || t == at::kChar || t == at::kInt || t == at::kBool,
      "Data type is unsupported (" << t << ")");
  switch (t) {
    case at::kHalf:
      value = DataType::kHalf;
      break;
    case at::kChar:
      value = DataType::kChar;
      break;
    case at::kInt:
      value = DataType::kInt;
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

TensorFormat::TensorFormat(at::MemoryFormat t) {
  TORCHTRT_CHECK(
      t == at::MemoryFormat::Contiguous || t == at::MemoryFormat::ChannelsLast,
      "Tensor format is unsupported (" << t << ")");

  switch (t) {
    case at::MemoryFormat::ChannelsLast:
      value = TensorFormat::kChannelsLast;
    case at::MemoryFormat::Contiguous:
    default:
      value = TensorFormat::kContiguous;
      break;
  }
}

Device::DeviceType::DeviceType(c10::DeviceType t) {
  TORCHTRT_CHECK(t == at::kCUDA, "Device type when specified using torch device enum must be torch::kCUDA");
  value = DeviceType::kGPU;
}

/* ====== DEFINE INPUTS CLASS MEMBERS ======*/
Input::Input(std::vector<int64_t> shape, TensorFormat format) {
  this->opt_shape = shape;
  this->min_shape = shape;
  this->max_shape = shape;
  this->shape = shape;
  this->dtype = DataType::kUnknown;
  this->format = format;
  this->input_is_dynamic = false;
}

Input::Input(std::vector<int64_t> shape, DataType dtype, TensorFormat format) {
  this->opt_shape = shape;
  this->min_shape = shape;
  this->max_shape = shape;
  this->shape = shape;
  this->dtype = dtype;
  this->format = format;
  this->input_is_dynamic = false;
}

Input::Input(c10::IntArrayRef shape, TensorFormat format) {
  this->opt_shape = torch_tensorrt::core::util::toVec(shape);
  this->min_shape = torch_tensorrt::core::util::toVec(shape);
  this->max_shape = torch_tensorrt::core::util::toVec(shape);
  this->shape = torch_tensorrt::core::util::toVec(shape);
  this->dtype = DataType::kUnknown;
  this->format = format;
  this->input_is_dynamic = false;
}

Input::Input(c10::IntArrayRef shape, DataType dtype, TensorFormat format) {
  this->opt_shape = torch_tensorrt::core::util::toVec(shape);
  this->min_shape = torch_tensorrt::core::util::toVec(shape);
  this->max_shape = torch_tensorrt::core::util::toVec(shape);
  this->shape = torch_tensorrt::core::util::toVec(shape);
  this->dtype = dtype;
  this->format = format;
  this->input_is_dynamic = false;
}

Input::Input(
    std::vector<int64_t> min_shape,
    std::vector<int64_t> opt_shape,
    std::vector<int64_t> max_shape,
    TensorFormat format) {
  this->opt_shape = opt_shape;
  this->min_shape = min_shape;
  this->max_shape = max_shape;
  this->shape = torch_tensorrt::core::util::toVec(
      torch_tensorrt::core::ir::Input(this->min_shape, this->opt_shape, this->max_shape).input_shape);
  this->dtype = DataType::kUnknown;
  this->format = format;
  this->input_is_dynamic = true;
}

Input::Input(
    std::vector<int64_t> min_shape,
    std::vector<int64_t> opt_shape,
    std::vector<int64_t> max_shape,
    DataType dtype,
    TensorFormat format) {
  this->opt_shape = opt_shape;
  this->min_shape = min_shape;
  this->max_shape = max_shape;
  this->shape = torch_tensorrt::core::util::toVec(
      torch_tensorrt::core::ir::Input(this->min_shape, this->opt_shape, this->max_shape).input_shape);
  this->dtype = dtype;
  this->format = format;
  this->input_is_dynamic = true;
}

Input::Input(c10::IntArrayRef min_shape, c10::IntArrayRef opt_shape, c10::IntArrayRef max_shape, TensorFormat format) {
  this->opt_shape = torch_tensorrt::core::util::toVec(opt_shape);
  this->min_shape = torch_tensorrt::core::util::toVec(min_shape);
  this->max_shape = torch_tensorrt::core::util::toVec(max_shape);
  this->shape = torch_tensorrt::core::util::toVec(
      torch_tensorrt::core::ir::Input(this->min_shape, this->opt_shape, this->max_shape).input_shape);
  this->dtype = DataType::kUnknown;
  this->format = format;
  this->input_is_dynamic = true;
}

Input::Input(
    c10::IntArrayRef min_shape,
    c10::IntArrayRef opt_shape,
    c10::IntArrayRef max_shape,
    DataType dtype,
    TensorFormat format) {
  this->opt_shape = torch_tensorrt::core::util::toVec(opt_shape);
  this->min_shape = torch_tensorrt::core::util::toVec(min_shape);
  this->max_shape = torch_tensorrt::core::util::toVec(max_shape);
  this->shape = torch_tensorrt::core::util::toVec(
      torch_tensorrt::core::ir::Input(this->min_shape, this->opt_shape, this->max_shape).input_shape);
  this->dtype = dtype;
  this->format = format;
  this->input_is_dynamic = true;
}

Input::Input(at::Tensor tensor) {
  this->opt_shape = tensor.sizes().vec();
  this->min_shape = tensor.sizes().vec();
  this->max_shape = tensor.sizes().vec();
  this->shape = tensor.sizes().vec();
  this->dtype = tensor.scalar_type();
  TORCHTRT_ASSERT(
      tensor.is_contiguous(at::MemoryFormat::ChannelsLast) || tensor.is_contiguous(at::MemoryFormat::Contiguous),
      "Tensor does not have a supported contiguous memory format, supported formats are contiguous or channel_last");
  at::MemoryFormat frmt;
  if (tensor.is_contiguous(at::MemoryFormat::Contiguous)) {
    frmt = at::MemoryFormat::Contiguous;
  } else {
    frmt = at::MemoryFormat::ChannelsLast;
  }
  this->format = frmt;
  this->input_is_dynamic = false;
}

/* ==========================================*/

torch_tensorrt::core::ir::Input to_internal_input(Input& i) {
  return torch_tensorrt::core::ir::Input(
      i.min_shape,
      i.opt_shape,
      i.max_shape,
      toTRTDataType(i.dtype),
      toTRTTensorFormat(i.format),
      !(i.dtype == DataType::kUnknown));
}

std::vector<torch_tensorrt::core::ir::Input> to_vec_internal_inputs(std::vector<Input>& external) {
  std::vector<torch_tensorrt::core::ir::Input> internal;
  for (auto range : external) {
    internal.push_back(to_internal_input(range));
  }
  return internal;
}

torch_tensorrt::core::runtime::CudaDevice to_internal_cuda_device(Device device) {
  auto device_type = nvinfer1::DeviceType::kGPU;
  switch (device.device_type) {
    case Device::DeviceType::kDLA:
      device_type = nvinfer1::DeviceType::kDLA;
      break;
    case Device::DeviceType::kGPU:
    default:
      device_type = nvinfer1::DeviceType::kGPU;
  }
  return torch_tensorrt::core::runtime::CudaDevice(device.gpu_id, device_type);
}
} // namespace torch_tensorrt
