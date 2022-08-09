#pragma once

#include <iostream>
#include <map>
#include <vector>

#include "NvInfer.h"
#include "torch/csrc/jit/ir/ir.h"

namespace torch_tensorrt {
namespace core {
namespace ir {

struct Input : torch::CustomClassHolder {
  Input(){};
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
  int id;
};

// Add to spec
struct GraphInputs {
  GraphInputs(std::vector<ir::Input> inputs);
  GraphInputs(torch::jit::IValue& input_signature);
  torch::jit::IValue input_signature; // nested Input, full input spec
  std::vector<Input> inputs; // flattend Input
  std::vector<std::vector<Input>> collection_inputs; // only support two layer nesting, e.g. ((a, b), [c, d], e)
};

typedef std::pair<GraphInputs, torch::jit::IValue> GraphIO; // Graph input output mapping

using StaticParams = std::map<torch::jit::Value*, torch::jit::IValue>;
StaticParams get_static_params(c10::ArrayRef<torch::jit::Value*> inputs, std::vector<torch::jit::IValue> params);

using InputSpecMap = std::unordered_map<const torch::jit::Value*, Input>;
using CollectionInputSpecMap = std::unordered_map<const torch::jit::Value*, std::vector<Input>>;

std::vector<const torch::jit::Value*> get_tensor_inputs(
    std::shared_ptr<torch::jit::Graph>& g,
    StaticParams& static_params);
InputSpecMap associate_specs_with_inputs(
    std::shared_ptr<torch::jit::Graph>& g,
    std::vector<Input> specs,
    StaticParams& static_params);
CollectionInputSpecMap associate_specs_with_collection_inputs(
    std::shared_ptr<torch::jit::Graph>& g,
    ir::GraphInputs graph_inputs,
    StaticParams& static_params);
InputSpecMap pair_input_vals_with_specs(std::vector<const torch::jit::Value*> vals, std::vector<Input> specs);
CollectionInputSpecMap pair_input_vals_with_specs_collection(
    std::vector<const torch::jit::Value*> vals,
    std::vector<std::vector<Input>>& specs);
std::vector<const torch::jit::Value*> get_tensor_inputs(
    std::shared_ptr<torch::jit::Graph>& g,
    StaticParams& static_params);
std::vector<const torch::jit::Value*> get_collection_inputs(
    std::shared_ptr<torch::jit::Graph>& g,
    StaticParams& static_params);

using TypeMap = std::unordered_map<const torch::jit::Value*, c10::optional<at::ScalarType>>;
using CollectionTypeMap = std::unordered_map<const torch::jit::Value*, std::vector<c10::optional<at::ScalarType>>>;

c10::optional<at::ScalarType> get_value_first_calc_dtype_opt(torch::jit::Block* b, torch::jit::Value* in);
ir::TypeMap get_block_first_calc_dtypes_opt(torch::jit::Block* b);
ir::CollectionTypeMap get_block_first_calc_dtypes_opt_collection(torch::jit::Block* b);
} // namespace ir
} // namespace core
} // namespace torch_tensorrt
