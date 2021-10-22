#pragma once

#include <sstream>
#include <string>

#include "torch/csrc/jit/ir/ir.h"

namespace torch_tensorrt {
namespace core {
namespace util {

inline std::string node_info(const torch::jit::Node* n) {
  std::stringstream ss;
  ss << *n;
  std::string node_info = ss.str();
  node_info.erase(std::remove(node_info.begin(), node_info.end(), '\n'), node_info.end());
  return node_info;
}

inline std::string value_info(const torch::jit::Value* v) {
  std::stringstream ss;
  ss << node_info(v->node());
  std::string value_info = ss.str();
  value_info.erase(std::remove(value_info.begin(), value_info.end(), '\n'), value_info.end());
  return value_info;
}

inline std::string schema_info(const torch::jit::FunctionSchema* s) {
  std::stringstream ss;
  ss << *s;
  std::string schema_info = ss.str();
  schema_info.erase(std::remove(schema_info.begin(), schema_info.end(), '\n'), schema_info.end());
  return schema_info;
}

inline std::vector<int64_t> toVec(c10::IntArrayRef a) {
  std::vector<int64_t> arr;
  for (auto i : a) {
    arr.push_back(i);
  }
  return arr;
}

inline c10::FunctionSchema GenerateGraphSchema(std::string method_name, std::shared_ptr<torch::jit::Graph>& g) {
  std::vector<c10::Argument> args;
  for (auto in : g->inputs()) {
    args.push_back(c10::Argument(in->debugName(), in->type()));
  }

  std::vector<c10::Argument> returns;
  for (auto out : g->outputs()) {
    returns.push_back(c10::Argument(out->debugName(), out->type()));
  }

  return c10::FunctionSchema(method_name, method_name, args, returns);
}

inline std::string GetPyTorchSourceCode(const torch::jit::Node* n) {
  std::string source_code = n->sourceRange().str();
  return source_code;
}

} // namespace util
} // namespace core
} // namespace torch_tensorrt
