#pragma once

#include <ostream>
#include <vector>

#include "NvInfer.h"
#include "core/ir/ir.h"
#include "torch/csrc/jit/ir/ir.h"

namespace torch_tensorrt {
namespace core {
namespace partitioning {

struct SegmentedBlock {
 public:
  enum SegmentedBlockTarget {
    kTorch,
    kTensorRT,
  };

  static std::string target_to_str(SegmentedBlockTarget t) {
    if (t == SegmentedBlockTarget::kTorch) {
      return "Torch";
    } else {
      return "TensorRT";
    }
  }

  using BlockID = uint64_t;

  SegmentedBlock() = default;
  SegmentedBlock(SegmentedBlockTarget blk_target) : target_(blk_target), g_(std::make_shared<torch::jit::Graph>()) {}
  SegmentedBlock(SegmentedBlockTarget blk_target, const std::vector<torch::jit::Node*>& nodes);
  SegmentedBlock(SegmentedBlockTarget blk_target, std::shared_ptr<torch::jit::Graph> g) : target_(blk_target), g_(g) {}
  SegmentedBlock(BlockID id, SegmentedBlockTarget blk_target, const std::vector<torch::jit::Node*>& nodes);

  torch::jit::Value* getOrAddInputForValue(torch::jit::Value* v);
  std::vector<ir::Input> construct_inputs_spec() const;
  torch::jit::Node* cloneNode(torch::jit::Node* node);
  void appendNode(torch::jit::Node* n) {
    cloneNode(n);
  }
  void registerOutput(torch::jit::Value* raw_output);
  torch::jit::graph_node_list nodes() {
    return g_->nodes();
  }
  const std::vector<torch::jit::Node*>& raw_nodes() const {
    return nodes_;
  }
  torch::jit::Block* block() {
    return g_->block();
  }
  std::shared_ptr<torch::jit::Graph>& g() {
    return g_;
  }
  void update_graph(std::shared_ptr<torch::jit::Graph> new_g) {
    g_ = new_g;
  }
  c10::ArrayRef<torch::jit::Value*> inputs() {
    return g_->inputs();
  }
  c10::ArrayRef<torch::jit::Value*> outputs() {
    return g_->outputs();
  }
  const std::vector<torch::jit::Value*>& raw_inputs() const {
    return inputs_;
  }
  const std::vector<torch::jit::Value*>& raw_outputs() const {
    return outputs_;
  }
  void eraseInput(size_t i);
  void eraseOutput(size_t i);
  bool contain_raw_value(torch::jit::Value* input) const {
    return old_to_new_.count(input);
  }
  void register_inshapes(std::vector<std::vector<int64_t>>& in_shapes, const ir::ShapeMode& shape_mode) {
    if (shape_mode == ir::ShapeMode::kMIN) {
      min_shapes_ = in_shapes;
    } else if (shape_mode == ir::ShapeMode::kOPT) {
      opt_shapes_ = in_shapes;
    } else {
      max_shapes_ = in_shapes;
    }
  }
  const std::vector<std::vector<int64_t>> in_opt_shapes() const {
    return opt_shapes_;
  }
  const std::vector<std::vector<int64_t>> in_min_shapes() const {
    return min_shapes_;
  }
  const std::vector<std::vector<int64_t>> in_max_shapes() const {
    return max_shapes_;
  }
  void register_intypes(std::vector<at::ScalarType>& in_types) {
    in_types_ = in_types;
  }
  const std::vector<at::ScalarType>& in_types() const {
    return in_types_;
  }

  BlockID get_id() {
    return id_;
  }
  void update_id(BlockID new_id) {
    id_ = new_id;
  }
  void update_target(SegmentedBlockTarget new_target) {
    target_ = new_target;
  }
  enum SegmentedBlockTarget target() const {
    return target_;
  }

  bool do_not_merge(void) const {
    return do_not_merge_;
  }

  void do_not_merge(bool x) {
    do_not_merge_ = x;
  }

  friend std::ostream& operator<<(std::ostream& os, const SegmentedBlock& b);

 private:
  BlockID id_;
  SegmentedBlockTarget target_;
  std::vector<std::vector<int64_t>> min_shapes_;
  std::vector<std::vector<int64_t>> opt_shapes_;
  std::vector<std::vector<int64_t>> max_shapes_;
  std::vector<at::ScalarType> in_types_;
  std::vector<torch::jit::Value*> inputs_;
  std::vector<torch::jit::Value*> outputs_;
  std::vector<torch::jit::Node*> nodes_;
  std::shared_ptr<torch::jit::Graph> g_;
  std::unordered_map<torch::jit::Value*, torch::jit::Value*> old_to_new_;
  bool do_not_merge_ = false;
};

std::ostream& operator<<(std::ostream& os, const SegmentedBlock::SegmentedBlockTarget& t);

} // namespace partitioning
} // namespace core
} // namespace torch_tensorrt
