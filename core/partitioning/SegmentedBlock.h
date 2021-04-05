#pragma once

#include <vector>

#include "core/conversion/conversion.h"
#include "torch/csrc/jit/ir/ir.h"

namespace trtorch {
namespace core {
namespace partitioning {

torch::jit::Value* getOrAddInputForValue(
    torch::jit::Value* old_value,
    torch::jit::Block* block,
    std::unordered_map<torch::jit::Value*, torch::jit::Value*>& old_to_new);

torch::jit::Node* cloneNode(
    torch::jit::Node* node,
    std::shared_ptr<torch::jit::Graph>& graph,
    std::unordered_map<torch::jit::Value*, torch::jit::Value*>& old_to_new);

struct SegmentedBlock {
 public:
  enum SegmentedBlockTarget {
    kTorch,
    kTensorRT,
  };

  SegmentedBlock() = default;

  SegmentedBlock(SegmentedBlockTarget blk_target) : target_(blk_target), g_(std::make_shared<torch::jit::Graph>()) {}

  SegmentedBlock(SegmentedBlockTarget blk_target, std::vector<torch::jit::Node*>& nodes)
      : target_(blk_target), g_(std::make_shared<torch::jit::Graph>()) {
    for (auto& node : nodes) {
      nodes_.push_back(node);
      appendNode(node);
    }
    registerInputs();
  }

  SegmentedBlock(SegmentedBlockTarget blk_target, std::shared_ptr<torch::jit::Graph> g) : target_(blk_target), g_(g) {}

  enum SegmentedBlockTarget target() {
    return target_;
  }

  void appendNode(torch::jit::Node* n) {
    cloneNode(n, g_, old_to_new_);
  }

  void registerInputs() {
    for (auto& value : g_->inputs()) {
      inputs_.push_back(old_to_new_[value]);
    }
  }

  void registerOutput(torch::jit::Value* raw_output) {
    outputs_.push_back(raw_output);
    g_->registerOutput(old_to_new_[raw_output]);
  }

  torch::jit::Block* block() {
    return g_->block();
  }

  c10::ArrayRef<torch::jit::Value*> inputs() {
    return g_->inputs();
  }

  void eraseInput(size_t i) {
    inputs_.erase(inputs_.begin() + i);
    g_->eraseInput(i);
  }

  c10::ArrayRef<torch::jit::Value*> outputs() {
    return g_->outputs();
  }

  void eraseOutput(size_t i) {
    outputs_.erase(outputs_.begin() + i);
    g_->eraseOutput(i);
  }

  const std::vector<torch::jit::Value*>& raw_inputs() const {
    return inputs_;
  }

  const std::vector<torch::jit::Value*>& raw_outputs() const {
    return outputs_;
  }

  const std::vector<torch::jit::Node*>& raw_nodes() const {
    return nodes_;
  }

  bool contain_raw_value(torch::jit::Value* input) {
    return old_to_new_.count(input);
  }

  torch::jit::graph_node_list nodes() {
    return g_->nodes();
  }

  void register_inshape(std::vector<nvinfer1::Dims>& in_shape) {
    in_shape_ = in_shape;
  }

  const std::vector<nvinfer1::Dims>& in_shape() const {
    return in_shape_;
  }

  std::shared_ptr<torch::jit::Graph>& g() {
    return g_;
  }

  void update_graph(std::shared_ptr<torch::jit::Graph> new_g) {
    g_ = new_g;
  }

  void update_target(SegmentedBlockTarget new_target) {
    target_ = new_target;
  }

  void update_mapping(std::unordered_map<torch::jit::Value*, torch::jit::Value*> &new_map) {
    old_to_new_ = new_map;
  }

// private:
  SegmentedBlockTarget target_;
  std::vector<nvinfer1::Dims> in_shape_;
  std::vector<torch::jit::Value*> inputs_;
  std::vector<torch::jit::Value*> outputs_;
  std::vector<torch::jit::Node*> nodes_;
  std::shared_ptr<torch::jit::Graph> g_;
  std::string trt_engine;
  std::unordered_map<torch::jit::Value*, torch::jit::Value*> old_to_new_;
};

//std::vector<SegmentedBlock> segment_graph(
//    std::shared_ptr<torch::jit::Graph> g,
//    const conversion::TorchFallback& fallback_info);
std::vector<SegmentedBlock> segment_graph(
    torch::jit::Block* block,
    const conversion::TorchFallback& fallback_info);


} // namespace partitioning
} // namespace core
} // namespace trtorch