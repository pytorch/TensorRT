#pragma once

#include "NvInfer.h"
#include "third_party/args/args.hpp"
#include "torch/script.h"
#include "torch/torch.h"

namespace torchtrtc {
namespace luts {

at::ScalarType to_torch_dtype(torchtrt::DataType dtype) {
  switch (dtype) {
    case torchtrt::DataType::kHalf:
      return at::kHalf;
    case torchtrt::DataType::kChar:
      return at::kChar;
    case torchtrt::DataType::kInt:
      return at::kInt;
    case torchtrt::DataType::kBool:
      return at::kBool;
    case torchtrt::DataType::kFloat:
    default:
      return at::kFloat;
  }
}

const std::unordered_map<nvinfer1::DataType, at::ScalarType>& get_trt_at_type_map() {
  static const std::unordered_map<nvinfer1::DataType, at::ScalarType> trt_at_type_map = {
      {nvinfer1::DataType::kFLOAT, at::kFloat},
      {nvinfer1::DataType::kHALF, at::kHalf},
      {nvinfer1::DataType::kINT32, at::kInt},
      {nvinfer1::DataType::kINT8, at::kChar},
      {nvinfer1::DataType::kBOOL, at::kBool},
  };
  return trt_at_type_map;
}

} // namespace luts
} // namespace torchtrtc