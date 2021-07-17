#pragma once

#include <vector>
#include <iostream>
#include "NvInfer.h"

namespace trtorch {
namespace core {
namespace ir {

struct Input {
  //Input(std::vector<int64_t> shape);
  //Input(std::vector<int64_t> min_shape, std::vector<int64_t> opt_shape, std::vector<int64_t> max_shape);
  Input(std::vector<int64_t> shape, nvinfer1::DataType dtype=nvinfer1::DataType::kFLOAT, nvinfer1::TensorFormat format=nvinfer1::TensorFormat::kLINEAR);
  Input(std::vector<int64_t> min_shape, std::vector<int64_t> opt_shape, std::vector<int64_t> max_shape, nvinfer1::DataType dtype=nvinfer1::DataType::kFLOAT, nvinfer1::TensorFormat format=nvinfer1::TensorFormat::kLINEAR);
  friend std::ostream& operator<<(std::ostream& os, const Input& input);

  bool input_is_dynamic = false;
  nvinfer1::Dims input_shape;
  nvinfer1::Dims min;
  nvinfer1::Dims max;
  nvinfer1::Dims opt;
  nvinfer1::DataType dtype;
  nvinfer1::TensorFormat format;
};

} // namespace ir
} // namespace core
} // namespace trtorch
