#include "ATen/ATen.h"
#include "torch/csrc/jit/api/module.h"
#include "torch/csrc/jit/ir/ir_views.h"

#include "core/partitioning/partitioning.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace partitioning {

void addSegmentedBlockToGraph(
    std::shared_ptr<torch::jit::Graph>& g,
    partitioning::SegmentedBlock& seg,
    std::unordered_map<torch::jit::Value*, torch::jit::Value*>& old_to_new_g) {
  // old_to_new_g contains: original global graph value => new global graph value,
  // mini_to_new_g: mini graph value -> new graph value
  std::unordered_map<torch::jit::Value*, torch::jit::Value*> mini_to_new_g;
  size_t input_idx = 0;
  if (seg.target() == partitioning::SegmentedBlock::kTensorRT && g->inputs().size() > 0) {
    if (g->inputs()[0]->type()->str().find("__torch__") == std::string::npos) {
      auto self = g->insertInput(0, "self_1");
      self->setType(seg.inputs()[0]->type());
    }
    mini_to_new_g[seg.inputs()[input_idx++]] = g->inputs()[0];
  }

  for (auto& raw_input : seg.raw_inputs()) {
    if (old_to_new_g.count(raw_input)) {
      mini_to_new_g[seg.inputs()[input_idx++]] = old_to_new_g[raw_input];
    }
  }

  for (const auto n : seg.nodes()) {
    util::cloneNode(n, g, mini_to_new_g);
  }

  // original graph value => new global graph value
  for (size_t i = 0; i < seg.raw_outputs().size(); ++i) {
    old_to_new_g[seg.raw_outputs()[i]] = mini_to_new_g[seg.outputs()[i]];
  }
  size_t offset = seg.target() == partitioning::SegmentedBlock::kTensorRT ? 1 : 0;
  for (size_t i = 0; i < seg.raw_inputs().size(); ++i) {
    if (!old_to_new_g.count(seg.raw_inputs()[i])) {
      old_to_new_g[seg.raw_inputs()[i]] = mini_to_new_g[seg.inputs()[i + offset]];
    }
  }

  return;
}

void addIfBlockToGraph(
    std::shared_ptr<torch::jit::Graph>& new_g,
    torch::jit::Node* if_node,
    const std::vector<GraphAndMapping>& graph_and_mappings,
    std::unordered_map<torch::jit::Value*, torch::jit::Value*>& old_to_new_g) {
  torch::jit::IfView if_view(if_node);

  // create a new if node in new_g and add corresponding inputs
  auto new_if = new_g->insertNode(new_g->create(torch::jit::prim::If, {}, 0));
  new_if->addInput(util::getOrAddInputForValue(if_view.cond(), new_g, old_to_new_g));

  // iterate over all blocks and add them to new created prim::If
  for (auto graph_and_mapping : graph_and_mappings) {
    auto new_if_block = new_if->addBlock();
    auto cur_block_graph = graph_and_mapping.first;
    auto cur_block_mapping = graph_and_mapping.second;
    std::unordered_map<torch::jit::Value*, torch::jit::Value*> block_graph_to_new_g;
    for (auto& i : cur_block_mapping) {
      // for every pair in then_mapping, old_value => mini graph value, if old_value also appears in old_to_new_g, then
      // it's mini graph's input
      if (old_to_new_g.count(i.first)) {
        block_graph_to_new_g[i.second] = old_to_new_g[i.first];
      }
    }

    auto env = [&](torch::jit::Value* v) { return util::getOrAddInputForValue(v, new_g, block_graph_to_new_g); };
    new_if_block->cloneFrom(cur_block_graph->block(), env);
    if (cur_block_graph->inputs().size() &&
        cur_block_graph->inputs()[0]->type()->str().find("__torch__") != std::string::npos) {
      if (new_g->inputs()[0]->type()->str().find("__torch__") == std::string::npos) {
        auto self = new_g->insertInput(0, "self_1");
        self->setType(cur_block_graph->inputs()[0]->type());
      }
      block_graph_to_new_g[cur_block_graph->inputs()[0]] = new_g->inputs()[0];
    }
    for (int i = cur_block_graph->inputs().size() - 1; i >= 0; --i) {
      new_if_block->inputs()[i]->replaceAllUsesWith(block_graph_to_new_g[cur_block_graph->inputs()[i]]);
      new_if_block->eraseInput(i);
    }
  }
  for (auto ov : if_view.outputs()) {
    auto no = new_if->addOutput();
    old_to_new_g[ov] = no;
    no->copyMetadata(ov);
  }
  return;
}

GraphAndMapping stitch(PartitioningCtx* ctx, torch::jit::Block* block) {
  auto new_g = std::make_shared<torch::jit::Graph>();

  // the mapping from lowering graph => fallback global graph
  std::unordered_map<torch::jit::Value*, torch::jit::Value*> old_to_new_g;
  for (auto input : block->inputs()) {
    util::getOrAddInputForValue(input, new_g, old_to_new_g);
  }

  for (auto seg_block : ctx->partitioned_blocks[block]) {
    LOG_INFO("Block segment:" << seg_block);
    if (seg_block.target() == partitioning::SegmentedBlock::kTensorRT) {
      addSegmentedBlockToGraph(new_g, seg_block, old_to_new_g);
    } else {
      if (seg_block.raw_nodes()[0]->kind() == torch::jit::prim::If) {
        auto if_node = seg_block.raw_nodes()[0];

        // convert the 2 blocks in prim::if and get the converted graph with mappings
        std::vector<GraphAndMapping> graph_and_mappings;
        for (auto cur_block : if_node->blocks()) {
          graph_and_mappings.push_back(stitch(ctx, cur_block));
        }
        addIfBlockToGraph(new_g, if_node, graph_and_mappings, old_to_new_g);

      } else {
        addSegmentedBlockToGraph(new_g, seg_block, old_to_new_g);
      }
    }
  }

  if (block->outputs().size() > 1) {
    std::vector<torch::jit::Value*> fallback_graph_vector;
    for (auto& output : block->outputs()) {
      if (old_to_new_g.count(output)) {
        fallback_graph_vector.push_back(old_to_new_g[output]);
      }
    }
    torch::jit::ArrayRef<torch::jit::Value*> fallback_graph_outputs(fallback_graph_vector);
    auto return_tuple_node = new_g->createTuple(fallback_graph_outputs);
    new_g->block()->appendNode(return_tuple_node);
    // Set the output as the produced tuple
    new_g->registerOutput(return_tuple_node->outputs()[0]);
  } else {
    if (block->outputs().size() && old_to_new_g.count(block->outputs()[0])) {
      new_g->registerOutput(old_to_new_g[block->outputs()[0]]);
    }
  }
  return {new_g, old_to_new_g};
}
} // namespace partitioning
} // namespace core
} // namespace torch_tensorrt
