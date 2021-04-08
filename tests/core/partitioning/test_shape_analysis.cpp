#include <string>
#include "core/lowering/lowering.h"
#include "core/partitioning/partitioning.h"
#include "core/util/trt_util.h"
#include "gtest/gtest.h"
#include "torch/script.h"

bool checkSegmentedBlockInputShape(
    std::vector<trtorch::core::partitioning::SegmentedBlock>& segmented_blocks,
    std::vector<std::vector<std::vector<int>>> in_shape) {
  if (segmented_blocks.size() != in_shape.size())
    return false;
  for (size_t i = 0; i < segmented_blocks.size(); ++i) {
    auto cur_block_in_shapes = segmented_blocks[i].in_shape();
    if (cur_block_in_shapes.size() != in_shape[i].size())
      return false;
    for (size_t j = 0; j < cur_block_in_shapes.size(); ++j) {
      auto cur_input_shape = trtorch::core::util::toVec(cur_block_in_shapes[j]);
      for (size_t k = 0; k < cur_input_shape.size(); ++k) {
        if (cur_input_shape[k] != in_shape[i][j][k])
          return false;
      }
    }
  }
  return true;
}

TEST(Partitioning, InferSegmentedBlockShapeCorrectly) {
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
  std::vector<trtorch::core::ir::InputRange> input_ranges{trtorch::core::ir::InputRange({3, 3, 16, 16})};

  std::vector<trtorch::core::partitioning::SegmentedBlock> segmented_blocks =
      trtorch::core::partitioning::Partition(g, input_ranges, partition_info);
  ASSERT_TRUE(
      checkSegmentedBlockInputShape(segmented_blocks, {{{3, 3, 16, 16}}, {{3, 16, 16, 16}}, {{3, 16, 16, 16}}}));
}

TEST(Partitioning, InferSegmentedBlockShapeCorrectlyEdge) {
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
  std::vector<trtorch::core::ir::InputRange> input_ranges{trtorch::core::ir::InputRange({3, 3, 16, 16})};

  std::vector<trtorch::core::partitioning::SegmentedBlock> segmented_blocks =
      trtorch::core::partitioning::Partition(g, input_ranges, partition_info);
  ASSERT_TRUE(checkSegmentedBlockInputShape(
      segmented_blocks, {{{3, 3, 16, 16}}, {{3, 32, 16, 16}}, {{3, 32, 16, 16}, {3, 16, 16, 16}}}));
}
