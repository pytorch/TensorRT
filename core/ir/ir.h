#pragma once

#include <iostream>
#include <map>
#include <vector>

#include "NvInfer.h"
#include "torch/csrc/jit/ir/ir.h"

namespace torch_tensorrt {
namespace core {
namespace ir {

struct Input {
  // Input(std::vector<int64_t> shape);
  // Input(std::vector<int64_t> min_shape, std::vector<int64_t> opt_shape, std::vector<int64_t> max_shape);
  Input(
      std::vector<int64_t> shape,
      nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT,
      nvinfer1::TensorFormat format = nvinfer1::TensorFormat::kLINEAR,
      bool dtype_is_user_defined = false);
  Input(
      std::vector<int64_t> min_shape,
      std::vector<int64_t> opt_shape,
      std::vector<int64_t> max_shape,
      nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT,
      nvinfer1::TensorFormat format = nvinfer1::TensorFormat::kLINEAR,
      bool dtype_is_used_defined = false);
  friend std::ostream& operator<<(std::ostream& os, const Input& input);

  bool input_is_dynamic = false;
  bool dtype_is_user_defined = false;
  nvinfer1::Dims input_shape;
  nvinfer1::Dims min;
  nvinfer1::Dims max;
  nvinfer1::Dims opt;
  nvinfer1::DataType dtype;
  nvinfer1::TensorFormat format;
};

using StaticParams = std::map<torch::jit::Value*, torch::jit::IValue>;
StaticParams get_static_params(c10::ArrayRef<torch::jit::Value*> inputs, std::vector<torch::jit::IValue> params);

using InputSpecMap = std::unordered_map<const torch::jit::Value*, Input>;

InputSpecMap associate_specs_with_inputs(
    std::shared_ptr<torch::jit::Graph>& g,
    std::vector<Input> specs,
    StaticParams& static_params);
InputSpecMap pair_input_vals_with_specs(std::vector<const torch::jit::Value*> vals, std::vector<Input> specs);
std::vector<const torch::jit::Value*> get_tensor_inputs(
    std::shared_ptr<torch::jit::Graph>& g,
    StaticParams& static_params);

using TypeMap = std::unordered_map<const torch::jit::Value*, c10::optional<at::ScalarType>>;

c10::optional<at::ScalarType> get_value_first_calc_dtype_opt(torch::jit::Block* b, torch::jit::Value* in);
ir::TypeMap get_block_first_calc_dtypes_opt(torch::jit::Block* b);

} // namespace ir
} // namespace core
} // namespace torch_tensorrt
