#include "core/ir/ir.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace ir {

bool valid_dtype_format_combo(nvinfer1::DataType dtype, nvinfer1::TensorFormat format) {
  switch (dtype) {
    case nvinfer1::DataType::kINT8: // Supports just Linear (NCHW)
      switch (format) {
        case nvinfer1::TensorFormat::kLINEAR:
          return true;
        case nvinfer1::TensorFormat::kHWC:
        default:
          return false;
      }
    case nvinfer1::DataType::kINT32: // Supports just Linear (NCHW)
      switch (format) {
        case nvinfer1::TensorFormat::kLINEAR:
          return true;
        case nvinfer1::TensorFormat::kHWC:
        default:
          return false;
      }
    case nvinfer1::DataType::kHALF: // Supports just Linear (NCHW)
      switch (format) {
        case nvinfer1::TensorFormat::kLINEAR:
          return true;
        case nvinfer1::TensorFormat::kHWC:
        default:
          return false;
      }
    case nvinfer1::DataType::kFLOAT: // Supports both Linear (NCHW) and channel last (NHWC)
      switch (format) {
        case nvinfer1::TensorFormat::kLINEAR:
          return true;
        case nvinfer1::TensorFormat::kHWC:
          return true;
        default:
          return false;
      }
    case nvinfer1::DataType::kBOOL: // Supports Linear (NCHW)
      switch (format) {
        case nvinfer1::TensorFormat::kLINEAR:
          return true;
        default:
          return false;
      }
    default:
      return false;
  }
}

bool valid_input_dtype(nvinfer1::DataType dtype) {
  switch (dtype) {
    case nvinfer1::DataType::kBOOL:
      return true;
    case nvinfer1::DataType::kFLOAT:
      return true;
    case nvinfer1::DataType::kHALF:
      return true;
    case nvinfer1::DataType::kINT8:
      return true;
    case nvinfer1::DataType::kINT32:
      return true;
    default:
      return false;
  }
}

Input::Input(
    std::vector<int64_t> shape,
    nvinfer1::DataType dtype,
    nvinfer1::TensorFormat format,
    bool dtype_is_user_defined) {
  if (shape.size() > 5) {
    LOG_WARNING("Verify that this dim size is accepted");
  }

  opt = util::toDims(shape);
  min = util::toDims(shape);
  max = util::toDims(shape);
  input_shape = util::toDims(shape);
  input_is_dynamic = false;

  TORCHTRT_CHECK(valid_input_dtype(dtype), "Unsupported input data type: " << dtype);
  this->dtype = dtype;
  TORCHTRT_CHECK(
      valid_dtype_format_combo(dtype, format),
      "Unsupported combination of dtype and tensor format: ("
          << dtype << ", " << format
          << "), Torch-TensorRT only supports contiguous format (NCHW) except with input type Float32 where channel last (NHWC) is also supported");
  this->format = format;
  this->dtype_is_user_defined = dtype_is_user_defined;
}

Input::Input(
    std::vector<int64_t> min_shape,
    std::vector<int64_t> opt_shape,
    std::vector<int64_t> max_shape,
    nvinfer1::DataType dtype,
    nvinfer1::TensorFormat format,
    bool dtype_is_user_defined) {
  if (min_shape.size() > 5 || opt_shape.size() > 5 || max_shape.size() > 5) {
    LOG_WARNING("Verify that this dim size is accepted");
  }

  std::set<size_t> sizes;
  sizes.insert(min_shape.size());
  sizes.insert(opt_shape.size());
  sizes.insert(max_shape.size());

  if (sizes.size() != 1) {
    LOG_ERROR(
        "Expected all input sizes have the same dimensions, but found dimensions: min("
        << min_shape.size() << "), opt(" << opt_shape.size() << "), max(" << max_shape.size() << ")");
  }

  min = util::toDims(min_shape);
  opt = util::toDims(opt_shape);
  max = util::toDims(max_shape);

  std::vector<int64_t> dyn_shape;
  for (size_t i = 0; i < opt_shape.size(); i++) {
    std::set<uint64_t> dim;
    dim.insert(min_shape[i]);
    dim.insert(opt_shape[i]);
    dim.insert(max_shape[i]);
    if (dim.size() != 1) {
      dyn_shape.push_back(-1);
      input_is_dynamic = true;
    } else {
      dyn_shape.push_back(opt_shape[i]);
    }
  }

  input_shape = util::toDims(dyn_shape);

  TORCHTRT_CHECK(valid_input_dtype(dtype), "Unsupported input data type: " << dtype);
  this->dtype = dtype;
  TORCHTRT_CHECK(
      valid_dtype_format_combo(dtype, format),
      "Unsupported combination of dtype and tensor format: ("
          << dtype << ", " << format
          << "), Torch-TensorRT only supports contiguous format (NCHW) except with input type Float32 where channel last (NHWC) is also supported");
  this->format = format;
  this->dtype_is_user_defined = dtype_is_user_defined;
}

std::ostream& operator<<(std::ostream& os, const Input& input) {
  if (!input.input_is_dynamic) {
    os << "Input(shape: " << input.input_shape << ", dtype: " << input.dtype << ", format: " << input.format << ')';
  } else {
    os << "Input(shape: " << input.input_shape << ", min: " << input.min << ", opt: " << input.opt
       << ", max: " << input.max << ", dtype: " << input.dtype << ", format: " << input.format << ')';
  }
  return os;
}

} // namespace ir
} // namespace core
} // namespace torch_tensorrt
