#pragma once

#include <vector>
#include "NvInfer.h"

namespace trtorch {
namespace core {
namespace ir {

struct InputRange {
  nvinfer1::Dims min;
  nvinfer1::Dims max;
  nvinfer1::Dims opt;
  nvinfer1::Dims input_shape;
  bool input_is_dynamic = false;
  // Should we restrict to unsigned?
  InputRange(std::vector<int64_t> d);
  InputRange(std::vector<int64_t> min_shape, std::vector<int64_t> opt_shape, std::vector<int64_t> max_shape);
};

} // namespace ir
} // namespace core
} // namespace trtorch