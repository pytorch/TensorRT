#pragma once

#include <vector>

#include "core/conversion/conversion.h"
#include "torch/csrc/jit/ir/ir.h"


namespace trtorch {
namespace core {
namespace partitioning {

torch::jit::Node* cloneNode(torch::jit::Node* node, std::shared_ptr<torch::jit::Graph> &graph,
                            std::unordered_map<torch::jit::Value*, torch::jit::Value*> &old_to_new);

struct SegmentedBlock {
 public:
  enum SegmentedBlockTarget {
    kTorch,
    kTensorRT,
  };

  SegmentedBlock(SegmentedBlockTarget blk_target) : target(blk_target), g_(std::make_shared<torch::jit::Graph>()) {}

  SegmentedBlock(SegmentedBlockTarget blk_target, std::shared_ptr<torch::jit::Graph> g) : target(blk_target), g_(g) {}

  void appendNode(torch::jit::Node* n) {
    last_node = cloneNode(n, g_, old_to_new_);
  }

  void registerOutput() {
    for (auto &value : last_node->outputs()) {
      g_->registerOutput(value);
    }
  }

  torch::jit::Block* block() {
    return g_->block();
  }

  c10::ArrayRef<torch::jit::Value*> inputs() {
    return g_->inputs();
  }

  torch::jit::graph_node_list nodes() {
    return g_->nodes();
  }

  void register_inshape(std::vector<nvinfer1::Dims>& in_shape) {
    in_shape_ = in_shape;
  }

  void register_outshape(std::vector<nvinfer1::Dims>& out_shape) {
    out_shape_ = out_shape;
  }

  SegmentedBlockTarget target;
  std::vector<nvinfer1::Dims> in_shape_;
  std::vector<nvinfer1::Dims> out_shape_;
//  std::vector<torch::jit::Value*> inputs_;
//  std::vector<torch::jit::Value*> outputs_;
  std::shared_ptr<torch::jit::Graph> g_;
  std::string trt_engine;
  torch::jit::Node* last_node;
  std::unordered_map<torch::jit::Value*, torch::jit::Value*> old_to_new_;

};

std::vector<SegmentedBlock> segment_graph(std::shared_ptr<torch::jit::Graph> g, std::vector<conversion::InputRange>& input_ranges);

}
}
}