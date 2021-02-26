#pragma once

#include <vector>

#include "core/conversion/conversion.h"
#include "torch/csrc/jit/ir/ir.h"

namespace trtorch {
namespace core {
namespace partitioning {

struct MiniGraph {
  MiniGraph() : graph_(std::make_shared<torch::jit::Graph>()) {}

  torch::jit::Value* addNewInputForValue(torch::jit::Value* old_value) {
    auto node = old_value->node();

    if (node->kind() == torch::jit::prim::Constant) {
      auto new_const = graph_->createClone(node, {nullptr});
      graph_->block()->prependNode(new_const);
      return new_const->output();
    }

    auto new_value = graph_->block()->addInput();
    return mapValueAndCopyMetadata(old_value, new_value);
  }

  torch::jit::Value* mapValueAndCopyMetadata(torch::jit::Value* old_value, torch::jit::Value* new_value) {
    this->old_to_new_[old_value] = new_value;
    new_value->copyMetadata(old_value);
    return new_value;
  }

  torch::jit::Value* getOrAddInputForValue(torch::jit::Value* v) {
    if (this->old_to_new_.count(v) == 0) {
      return addNewInputForValue(v);
    } else {
      return this->old_to_new_[v];
    }
  }

  torch::jit::Node* cloneNode(torch::jit::Node* node) {
    auto* block = graph_->block();
    auto env = [this](torch::jit::Value* v) { return getOrAddInputForValue(v); };

    last_node = node;
    auto new_node = block->appendNode(graph_->createClone(node, env));
    for (size_t i = 0; i < node->outputs().size(); ++i) {
      auto oo = node->outputs()[i];
      auto no = new_node->outputs()[i];
      old_to_new_[oo] = no;
    }

    return new_node;
  }

  void registerOutput() {
    for (auto &value : last_node->outputs()) {
      graph_->registerOutput(getOrAddInputForValue(value));
    }
  }

  std::shared_ptr<torch::jit::Graph> graph_;
  torch::jit::Node* last_node;
  std::unordered_map<torch::jit::Value*, torch::jit::Value*> old_to_new_;
};

struct SegmentedBlock {
 public:
  enum SegmentedBlockTarget {
    kTorch,
    kTensorRT,
  };

  SegmentedBlock(SegmentedBlockTarget blk_target) : target(blk_target) {}

  void appendNode(torch::jit::Node* n) {
    mini_g.cloneNode(n);
  }

  void registerOutput() {
    mini_g.registerOutput();
  }


  SegmentedBlockTarget target;
  nvinfer1::Dims in_shape;
  nvinfer1::Dims out_shape;
  std::vector<torch::jit::Value*> inputs;
  std::vector<torch::jit::Value*> outputs;
  MiniGraph mini_g;
//  std::shared_ptr<torch::jit::Graph> g;
//  std::vector<torch::jit::Node*> nodes;
//  torch::jit::Block block;
  std::string trt_engine;
};

std::vector<SegmentedBlock> segment_graph(std::shared_ptr<torch::jit::Graph> g);

}
}
}