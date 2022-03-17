#include "core/conversion/converters/Weights.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {

Weights::Weights() {
  this->num_input_maps = 0;
  this->num_output_maps = 0;
  this->data.type = nvinfer1::DataType::kFLOAT;
  this->data.values = nullptr;
  this->data.count = 0;
}

Weights::Weights(ConversionCtx* ctx, float val) {
  this->num_input_maps = 1;
  this->num_output_maps = 1;

  this->data.type = nvinfer1::DataType::kFLOAT;
  float* buf = reinterpret_cast<float*>(malloc(1 * sizeof(float)));
  buf[0] = val;
  this->data.values = buf;
  this->data.count = 1;
  ctx->builder_resources.push_back(buf);

  this->shape.nbDims = 0;
  this->kernel_shape.nbDims = 0;
}

Weights::Weights(ConversionCtx* ctx, int32_t val) {
  this->num_input_maps = 1;
  this->num_output_maps = 1;

  this->data.type = nvinfer1::DataType::kINT32;
  int32_t* buf = reinterpret_cast<int32_t*>(malloc(1 * sizeof(int32_t)));
  buf[0] = val;
  this->data.values = buf;
  this->data.count = 1;
  ctx->builder_resources.push_back(buf);

  this->shape.nbDims = 0;
  this->kernel_shape.nbDims = 0;
}

Weights::Weights(ConversionCtx* ctx, at::Tensor t) {
  if (t.sizes().size() > nvinfer1::Dims::MAX_DIMS) {
    TORCHTRT_THROW_ERROR(
        "The tensor requested to be converted to nvinfer1::Weights exceeds the max number of dimensions for TensorRT");
  }
  this->shape = util::toDims(t.sizes());
  if (t.sizes().size() >= 2) {
    // Linear and Conv2/3D
    this->num_input_maps = t.sizes()[1];
    this->num_output_maps = t.sizes()[0];
  } else {
    // Bias
    this->num_input_maps = t.sizes()[0];
    this->num_output_maps = t.sizes()[0];
  }

  if (t.sizes().size() > 2) {
    this->kernel_shape.nbDims = t.sizes().size() - 2;

    for (size_t i = 2; i < t.sizes().size(); i++) {
      this->kernel_shape.d[i - 2] = t.sizes()[i];
      this->data.count *= this->kernel_shape.d[i - 2];
    }
  } else {
    this->kernel_shape.nbDims = 1;
    this->kernel_shape.d[0] = 1;
  }
  auto t_cpu = t.to(at::kCPU);
  t_cpu = t_cpu.contiguous();
  auto dtype_optional = util::optScalarTypeToTRTDataType(t_cpu.scalar_type());
  if (!dtype_optional) {
    TORCHTRT_THROW_ERROR(
        "The tensor requested to be converted to nvinfer1::Weights is of an unsupported type: "
        << dtype_optional.value());
  }

  // Store the data in the conversion context so it remains until building is
  // complete

  void* buf = nullptr;

  if (dtype_optional.value() == nvinfer1::DataType::kFLOAT) {
    buf = malloc(t_cpu.numel() * sizeof(float));
    memcpy(buf, t_cpu.data_ptr(), t_cpu.numel() * sizeof(float));
  } else if (dtype_optional.value() == nvinfer1::DataType::kHALF) {
    buf = malloc(t_cpu.numel() * (sizeof(float) / 2));
    memcpy(buf, t_cpu.data_ptr(), t_cpu.numel() * (sizeof(float) / 2));
  } else if (dtype_optional.value() == nvinfer1::DataType::kINT8) {
    buf = malloc(t_cpu.numel() * sizeof(char));
    memcpy(buf, t_cpu.data_ptr(), t_cpu.numel() * sizeof(char));
  } else if (dtype_optional.value() == nvinfer1::DataType::kINT32) {
    buf = malloc(t_cpu.numel() * sizeof(int));
    memcpy(buf, t_cpu.data_ptr(), t_cpu.numel() * sizeof(int));
  } else if (dtype_optional.value() == nvinfer1::DataType::kBOOL) {
    buf = malloc(t_cpu.numel() * sizeof(bool));
    memcpy(buf, t_cpu.data_ptr(), t_cpu.numel() * sizeof(bool));
  } else {
    TORCHTRT_THROW_ERROR("Found unsupported data type for tensor to weight conversion");
  }

  ctx->builder_resources.push_back(buf);

  this->data.type = dtype_optional.value();
  this->data.count = t_cpu.numel();
  this->data.values = buf;

  LOG_DEBUG(*this);
}

// clang-format off
std::ostream& operator<<(std::ostream& os, const Weights& w) {
  os << "Weights: " << w.shape
     << "\n    Data Type: " << w.data.type
     << "\n    Number of input maps: " << w.num_input_maps
     << "\n    Number of output maps: " << w.num_output_maps
     << "\n    Element shape: [";
  for (int i = 0; i < w.kernel_shape.nbDims; i++) {
    os << w.kernel_shape.d[i];
    if (i + 1 < w.kernel_shape.nbDims) {
      os << ',';
    }
  }
  os << ']';
  return os;
}
// clang-format on
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
