#include <string>
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/script.h"
#include "trtorch/trtorch.h"
#include "core/lowering/lowering.h"
#include "core/partitioning/partitioning.h"


bool checkSegmentedBlockNumber(std::vector<trtorch::core::partitioning::SegmentedBlock>& segmented_blocks,
                               trtorch::core::partitioning::SegmentedBlock::SegmentedBlockTarget target, int target_count) {
  for (auto &seg_block : segmented_blocks) {
    if (seg_block.target() == target) {
      target_count--;
    }
  }
  return target_count == 0;
}

bool checkSegmentedBlockNodesMapping(std::vector<trtorch::core::partitioning::SegmentedBlock>& segmented_blocks,
                                     std::shared_ptr<torch::jit::Graph> g, std::vector<std::vector<int>> nodes_index) {
  std::vector<torch::jit::Node*> graph_nodes;
  for (const auto n : g->nodes()) {
    if (n->kind() != torch::jit::prim::Constant) {
      graph_nodes.push_back(n);
    }
  }
  for (size_t i = 0; i < nodes_index.size(); ++i) {
    size_t seg_block_node_id = 0;
    for (int j : nodes_index[i]) {
      if (segmented_blocks[i].raw_nodes()[seg_block_node_id++] != graph_nodes[j]) {
        return false;
      }
    }
    if (seg_block_node_id != segmented_blocks[i].raw_nodes().size()) return false;
  }
  return true;
}

TEST(Partitioning, SegmentingGraphDefaultCorrectly) {
  torch::jit::script::Module mod;
  try {
  mod = torch::jit::load("tests/core/partitioning/test_base_model.jit");
  } catch (const c10::Error& e) {
  std::cerr << "error loading the model\n";
  return;
  }
  auto graph_and_parameters = trtorch::core::lowering::Lower(mod, "forward");
  auto g = graph_and_parameters.first;

  trtorch::core::partitioning::PartitionInfo partition_info;
  partition_info.enabled = true;
  std::vector<trtorch::core::partitioning::SegmentedBlock> segmented_blocks = trtorch::core::partitioning::segment_graph(g, partition_info);
  ASSERT_TRUE(checkSegmentedBlockNumber(segmented_blocks, trtorch::core::partitioning::SegmentedBlock::kTensorRT, 2));
  ASSERT_TRUE(checkSegmentedBlockNumber(segmented_blocks, trtorch::core::partitioning::SegmentedBlock::kTorch, 1));
  ASSERT_TRUE(checkSegmentedBlockNodesMapping(segmented_blocks, g, {{0, 1, 2}, {3}, {4}}));
}

TEST(Partitioning, SegmentingGraphWithMinBlockSizeCorrectly) {
  torch::jit::script::Module mod;
  try {
  mod = torch::jit::load("tests/core/partitioning/test_base_model.jit");
  } catch (const c10::Error& e) {
  std::cerr << "error loading the model\n";
  return;
  }
  auto graph_and_parameters = trtorch::core::lowering::Lower(mod, "forward");
  auto g = graph_and_parameters.first;

  trtorch::core::partitioning::PartitionInfo partition_info;
  partition_info.enabled = true;
  partition_info.min_block_size = 3;
  std::vector<trtorch::core::partitioning::SegmentedBlock> segmented_blocks = trtorch::core::partitioning::segment_graph(g, partition_info);
  ASSERT_TRUE(checkSegmentedBlockNumber(segmented_blocks, trtorch::core::partitioning::SegmentedBlock::kTensorRT, 1));
  ASSERT_TRUE(checkSegmentedBlockNumber(segmented_blocks, trtorch::core::partitioning::SegmentedBlock::kTorch, 1));
  ASSERT_TRUE(checkSegmentedBlockNodesMapping(segmented_blocks, g, {{0, 1, 2}, {3, 4}}));
}

TEST(Partitioning, SegmentingGraphWithForcedOPeCorrectly) {
  torch::jit::script::Module mod;
  try {
  mod = torch::jit::load("tests/core/partitioning/test_base_model.jit");
  } catch (const c10::Error& e) {
  std::cerr << "error loading the model\n";
  return;
  }
  auto graph_and_parameters = trtorch::core::lowering::Lower(mod, "forward");
  auto g = graph_and_parameters.first;

  trtorch::core::partitioning::PartitionInfo partition_info;
  partition_info.enabled = true;
  partition_info.forced_fallback_operators.push_back("aten::relu");
  std::vector<trtorch::core::partitioning::SegmentedBlock> segmented_blocks = trtorch::core::partitioning::segment_graph(g, partition_info);
  ASSERT_TRUE(checkSegmentedBlockNumber(segmented_blocks, trtorch::core::partitioning::SegmentedBlock::kTensorRT, 3));
  ASSERT_TRUE(checkSegmentedBlockNumber(segmented_blocks, trtorch::core::partitioning::SegmentedBlock::kTorch, 2));
  ASSERT_TRUE(checkSegmentedBlockNodesMapping(segmented_blocks, g, {{0}, {1}, {2}, {3}, {4}}));
}

TEST(Partitioning, SegmentingGraphDefaultCorrectlyEdge) {
  torch::jit::script::Module mod;
  try {
  mod = torch::jit::load("tests/core/partitioning/test_edge_model.jit");
  } catch (const c10::Error& e) {
  std::cerr << "error loading the model\n";
  return;
  }
  auto graph_and_parameters = trtorch::core::lowering::Lower(mod, "forward");
  auto g = graph_and_parameters.first;

  trtorch::core::partitioning::PartitionInfo partition_info;
  partition_info.enabled = true;
  std::vector<trtorch::core::partitioning::SegmentedBlock> segmented_blocks = trtorch::core::partitioning::segment_graph(g, partition_info);
  ASSERT_TRUE(checkSegmentedBlockNumber(segmented_blocks, trtorch::core::partitioning::SegmentedBlock::kTensorRT, 2));
  ASSERT_TRUE(checkSegmentedBlockNumber(segmented_blocks, trtorch::core::partitioning::SegmentedBlock::kTorch, 1));
  ASSERT_TRUE(checkSegmentedBlockNodesMapping(segmented_blocks, g, {{0, 1}, {2}, {3, 4, 5, 6}}));
}

TEST(Partitioning, SegmentingGraphWithMinBlockSizeCorrectlyEdge) {
  torch::jit::script::Module mod;
  try {
  mod = torch::jit::load("tests/core/partitioning/test_edge_model.jit");
  } catch (const c10::Error& e) {
  std::cerr << "error loading the model\n";
  return;
  }
  auto graph_and_parameters = trtorch::core::lowering::Lower(mod, "forward");
  auto g = graph_and_parameters.first;

  trtorch::core::partitioning::PartitionInfo partition_info;
  partition_info.enabled = true;
  partition_info.min_block_size = 3;
  std::vector<trtorch::core::partitioning::SegmentedBlock> segmented_blocks = trtorch::core::partitioning::segment_graph(g, partition_info);
  ASSERT_TRUE(checkSegmentedBlockNumber(segmented_blocks, trtorch::core::partitioning::SegmentedBlock::kTensorRT, 1));
  ASSERT_TRUE(checkSegmentedBlockNumber(segmented_blocks, trtorch::core::partitioning::SegmentedBlock::kTorch, 1));
  ASSERT_TRUE(checkSegmentedBlockNodesMapping(segmented_blocks, g, {{0, 1, 2}, {3, 4, 5, 6}}));
}

TEST(Partitioning, SegmentingGraphWithForcedOPeCorrectlyEdge) {
  torch::jit::script::Module mod;
  try {
  mod = torch::jit::load("tests/core/partitioning/test_edge_model.jit");
  } catch (const c10::Error& e) {
  std::cerr << "error loading the model\n";
  return;
  }
  auto graph_and_parameters = trtorch::core::lowering::Lower(mod, "forward");
  auto g = graph_and_parameters.first;

  trtorch::core::partitioning::PartitionInfo partition_info;
  partition_info.enabled = true;
  partition_info.forced_fallback_operators.push_back("aten::relu");
  std::vector<trtorch::core::partitioning::SegmentedBlock> segmented_blocks = trtorch::core::partitioning::segment_graph(g, partition_info);
  ASSERT_TRUE(checkSegmentedBlockNumber(segmented_blocks, trtorch::core::partitioning::SegmentedBlock::kTensorRT, 3));
  ASSERT_TRUE(checkSegmentedBlockNumber(segmented_blocks, trtorch::core::partitioning::SegmentedBlock::kTorch, 2));
  ASSERT_TRUE(checkSegmentedBlockNodesMapping(segmented_blocks, g, {{0, 1}, {2}, {3}, {4}, {5, 6}}));
}