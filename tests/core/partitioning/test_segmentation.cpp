#include <string>
#include "core/partitioning/partitioning.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/script.h"
#include "torch_tensorrt/torch_tensorrt.h"

bool checkSegmentedBlockNumber(
    torch_tensorrt::core::partitioning::PartitionedGraph& segmented_blocks,
    torch_tensorrt::core::partitioning::SegmentedBlock::SegmentedBlockTarget target,
    int target_count) {
  int64_t cnt = 0;
  for (auto& seg_block : segmented_blocks) {
    if (seg_block.target() == target) {
      cnt++;
    }
  }
  std::cout << "Found count of " << cnt << " " << target << " blocks (looking for " << target_count << " blocks)"
            << std::endl;

  if (target_count != cnt) {
    std::cout << segmented_blocks << std::endl;
  }

  return target_count == cnt;
}

bool checkSegmentedBlockNodesMapping(
    std::vector<torch_tensorrt::core::partitioning::SegmentedBlock>& segmented_blocks,
    std::shared_ptr<torch::jit::Graph> g,
    std::vector<std::vector<int>> nodes_index) {
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
    if (seg_block_node_id != segmented_blocks[i].raw_nodes().size())
      return false;
  }
  return true;
}

TEST(Partitioning, SegmentSequentialModelCorrectly) {
  const auto graph = R"IR(
        graph(%0 : Tensor,
              %w1 : Float(32, 3, 3, 3, strides=[27, 9, 3, 1]),
              %b1 : Float(32),
              %w2 : Float(16, 32, 3, 3, strides=[288, 9, 3, 1]),
              %b2 : Float(16),
              %w3 : Float(8, 16, 3, 3, strides=[144, 9, 3, 1]),
              %b3 : Float(8)):
          %2 : int[] = prim::Constant[value=[1, 1]]()
          %3 : int = prim::Constant[value=1]()
          %10 : bool = prim::Constant[value=0]()
          %11 : int[] = prim::Constant[value=[0, 0]]()
          %12: Tensor = aten::_convolution(%0, %w1, %b1, %2, %2, %2, %10, %11, %3, %10, %10, %10, %10)
          %13 : Tensor = aten::relu(%12)
          %14 : Tensor = aten::_convolution(%13, %w2, %b2, %2, %2, %2, %10, %11, %3, %10, %10, %10, %10)
          %15 : Tensor = aten::log_sigmoid(%14)
          %16 : Tensor = aten::_convolution(%15, %w3, %b3, %2, %2, %2, %10, %11, %3, %10, %10, %10, %10)
          return (%16))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  torch_tensorrt::core::partitioning::PartitionInfo partition_info;
  partition_info.enabled = true;
  std::unordered_map<torch::jit::Node*, int> fallback_nodes;
  std::vector<torch_tensorrt::core::partitioning::SegmentedBlock> segmented_blocks =
      torch_tensorrt::core::partitioning::segment_graph(g->block(), partition_info, fallback_nodes);
  ASSERT_TRUE(
      checkSegmentedBlockNumber(segmented_blocks, torch_tensorrt::core::partitioning::SegmentedBlock::kTensorRT, 2));
  ASSERT_TRUE(
      checkSegmentedBlockNumber(segmented_blocks, torch_tensorrt::core::partitioning::SegmentedBlock::kTorch, 1));
  ASSERT_TRUE(checkSegmentedBlockNodesMapping(segmented_blocks, g, {{0, 1, 2}, {3}, {4}}));
}

TEST(Partitioning, SegmentSequentialModelWithMinBlockSizeCorrectly) {
  const auto graph = R"IR(
          graph(%0 : Tensor,
                %w1 : Float(32, 3, 3, 3, strides=[27, 9, 3, 1]),
                %b1 : Float(32),
                %w2 : Float(16, 32, 3, 3, strides=[288, 9, 3, 1]),
                %b2 : Float(16),
                %w3 : Float(8, 16, 3, 3, strides=[144, 9, 3, 1]),
                %b3 : Float(8)):
            %2 : int[] = prim::Constant[value=[1, 1]]()
            %3 : int = prim::Constant[value=1]()
            %10 : bool = prim::Constant[value=0]()
            %11 : int[] = prim::Constant[value=[0, 0]]()
            %12: Tensor = aten::_convolution(%0, %w1, %b1, %2, %2, %2, %10, %11, %3, %10, %10, %10, %10)
            %13 : Tensor = aten::relu(%12)
            %14 : Tensor = aten::_convolution(%13, %w2, %b2, %2, %2, %2, %10, %11, %3, %10, %10, %10, %10)
            %15 : Tensor = aten::log_sigmoid(%14)
            %16 : Tensor = aten::_convolution(%15, %w3, %b3, %2, %2, %2, %10, %11, %3, %10, %10, %10, %10)
            return (%16))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  torch_tensorrt::core::partitioning::PartitionInfo partition_info;
  partition_info.enabled = true;
  partition_info.min_block_size = 3;
  std::unordered_map<torch::jit::Node*, int> fallback_nodes;
  std::vector<torch_tensorrt::core::partitioning::SegmentedBlock> segmented_blocks =
      torch_tensorrt::core::partitioning::segment_graph(g->block(), partition_info, fallback_nodes);
  ASSERT_TRUE(
      checkSegmentedBlockNumber(segmented_blocks, torch_tensorrt::core::partitioning::SegmentedBlock::kTensorRT, 1));
  ASSERT_TRUE(
      checkSegmentedBlockNumber(segmented_blocks, torch_tensorrt::core::partitioning::SegmentedBlock::kTorch, 1));
  ASSERT_TRUE(checkSegmentedBlockNodesMapping(segmented_blocks, g, {{0, 1, 2}, {3, 4}}));
}

TEST(Partitioning, SegmentModelWithMinBlockSizeCausedFallbackCorrectly) {
  const auto graph = R"IR(
          graph(%0 : Tensor,
                %1 : Tensor,
                %2 : Tensor):
            %3 : int[] = prim::Constant[value=[-1, 5]]()
            %4 : int[] = prim::Constant[value=[-1]]()
            %5 : int = prim::Constant[value=2]()
            %6 : int = prim::Constant[value=4]()
            %7 : int = prim::Constant[value=5]()
            %8 : int = prim::Constant[value=0]()
            %9 : bool = prim::Constant[value=0]()
            %10 : NoneType = prim::Constant()
            %11 : int = prim::Constant[value=1]()
            %12: Tensor = aten::reshape(%1, %4)
            %13: Tensor = aten::reshape(%2, %3)
            %14: Tensor = aten::reshape(%1, %3)
            %15 : Tensor = aten::to(%12, %6, %9, %9, %10)
            %16 : int = aten::size(%1, %8)
            %17 : int[] = prim::ListConstruct(%16, %6, %5, %7)
            %18 : Tensor = aten::index_add_(%14, %8, %15, %13, %11)
            %20 : Tensor = aten::reshape(%18, %17)
            return (%20))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  torch_tensorrt::core::partitioning::PartitionInfo partition_info;
  partition_info.enabled = true;
  partition_info.min_block_size = 3;
  std::unordered_map<torch::jit::Node*, int> fallback_nodes;
  std::vector<torch_tensorrt::core::partitioning::SegmentedBlock> segmented_blocks =
      torch_tensorrt::core::partitioning::segment_graph(g->block(), partition_info, fallback_nodes);
  ASSERT_TRUE(
      checkSegmentedBlockNumber(segmented_blocks, torch_tensorrt::core::partitioning::SegmentedBlock::kTensorRT, 1));
  ASSERT_TRUE(
      checkSegmentedBlockNumber(segmented_blocks, torch_tensorrt::core::partitioning::SegmentedBlock::kTorch, 1));
  ASSERT_TRUE(checkSegmentedBlockNodesMapping(segmented_blocks, g, {{0, 1, 2, 3}, {4, 5, 6, 7}}));
}

TEST(Partitioning, SegmentSequentialModelWithForcedOPCorrectly) {
  const auto graph = R"IR(
            graph(%0 : Tensor,
                  %w1 : Float(32, 3, 3, 3, strides=[27, 9, 3, 1]),
                  %b1 : Float(32),
                  %w2 : Float(16, 32, 3, 3, strides=[288, 9, 3, 1]),
                  %b2 : Float(16),
                  %w3 : Float(8, 16, 3, 3, strides=[144, 9, 3, 1]),
                  %b3 : Float(8)):
              %2 : int[] = prim::Constant[value=[1, 1]]()
              %3 : int = prim::Constant[value=1]()
              %10 : bool = prim::Constant[value=0]()
              %11 : int[] = prim::Constant[value=[0, 0]]()
              %12: Tensor = aten::_convolution(%0, %w1, %b1, %2, %2, %2, %10, %11, %3, %10, %10, %10, %10)
              %13 : Tensor = aten::relu(%12)
              %14 : Tensor = aten::_convolution(%13, %w2, %b2, %2, %2, %2, %10, %11, %3, %10, %10, %10, %10)
              %15 : Tensor = aten::log_sigmoid(%14)
              %16 : Tensor = aten::_convolution(%15, %w3, %b3, %2, %2, %2, %10, %11, %3, %10, %10, %10, %10)
              return (%16))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  torch_tensorrt::core::partitioning::PartitionInfo partition_info;
  partition_info.enabled = true;
  partition_info.forced_fallback_operators.push_back("aten::relu");
  std::unordered_map<torch::jit::Node*, int> fallback_nodes;
  std::vector<torch_tensorrt::core::partitioning::SegmentedBlock> segmented_blocks =
      torch_tensorrt::core::partitioning::segment_graph(g->block(), partition_info, fallback_nodes);
  ASSERT_TRUE(
      checkSegmentedBlockNumber(segmented_blocks, torch_tensorrt::core::partitioning::SegmentedBlock::kTensorRT, 3));
  ASSERT_TRUE(
      checkSegmentedBlockNumber(segmented_blocks, torch_tensorrt::core::partitioning::SegmentedBlock::kTorch, 2));
  ASSERT_TRUE(checkSegmentedBlockNodesMapping(segmented_blocks, g, {{0}, {1}, {2}, {3}, {4}}));
}

TEST(Partitioning, SegmentBranchModelCorrectly) {
  const auto graph = R"IR(
                graph(%0 : Tensor,
                      %1 : Float(32, 3, 3, 3, strides=[27, 9, 3, 1]),
                      %2 : Float(32),
                      %3 : Float(16, 32, 3, 3, strides=[288, 9, 3, 1]),
                      %4 : Float(16)):
                  %5 : int[] = prim::Constant[value=[0, 0]]()
                  %6 : int[] = prim::Constant[value=[2, 2]]()
                  %7 : bool = prim::Constant[value=0]()
                  %8 : int[] = prim::Constant[value=[1, 1]]()
                  %9 : int = prim::Constant[value=1]()
                  %10: Tensor = aten::_convolution(%0, %1, %2, %8, %8, %8, %7, %5, %9, %7, %7, %7, %7)
                  %11 : Tensor = aten::_convolution(%10, %3, %4, %8, %8, %8, %7, %5, %9, %7, %7, %7, %7)
                  %12: Tensor = aten::log_sigmoid(%10)
                  %13 : Tensor = aten::_convolution(%12, %3, %4,  %8, %8, %8, %7, %5, %9, %7, %7, %7, %7)
                  %14 : Tensor = aten::relu(%11)
                  %15 : Tensor = aten::add(%13, %14, %9)
                  %16 : Tensor = aten::max_pool2d(%15, %6, %6, %5, %8, %7)
                  return (%16))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  torch_tensorrt::core::partitioning::PartitionInfo partition_info;
  partition_info.enabled = true;
  std::unordered_map<torch::jit::Node*, int> fallback_nodes;
  std::vector<torch_tensorrt::core::partitioning::SegmentedBlock> segmented_blocks =
      torch_tensorrt::core::partitioning::segment_graph(g->block(), partition_info, fallback_nodes);
  ASSERT_TRUE(
      checkSegmentedBlockNumber(segmented_blocks, torch_tensorrt::core::partitioning::SegmentedBlock::kTensorRT, 2));
  ASSERT_TRUE(
      checkSegmentedBlockNumber(segmented_blocks, torch_tensorrt::core::partitioning::SegmentedBlock::kTorch, 1));
  ASSERT_TRUE(checkSegmentedBlockNodesMapping(segmented_blocks, g, {{0, 1}, {2}, {3, 4, 5, 6}}));
}

TEST(Partitioning, SegmentBranchModelWithMinBlockSizeCorrectly) {
  const auto graph = R"IR(
              graph(%0 : Tensor,
                    %1 : Float(32, 3, 3, 3, strides=[27, 9, 3, 1]),
                    %2 : Float(32),
                    %3 : Float(16, 32, 3, 3, strides=[288, 9, 3, 1]),
                    %4 : Float(16)):
                %5 : int[] = prim::Constant[value=[0, 0]]()
                %6 : int[] = prim::Constant[value=[2, 2]]()
                %7 : bool = prim::Constant[value=0]()
                %8 : int[] = prim::Constant[value=[1, 1]]()
                %9 : int = prim::Constant[value=1]()
                %10: Tensor = aten::_convolution(%0, %1, %2, %8, %8, %8, %7, %5, %9, %7, %7, %7, %7)
                %11 : Tensor = aten::_convolution(%10, %3, %4, %8, %8, %8, %7, %5, %9, %7, %7, %7, %7)
                %12: Tensor = aten::log_sigmoid(%10)
                %13 : Tensor = aten::_convolution(%12, %3, %4,  %8, %8, %8, %7, %5, %9, %7, %7, %7, %7)
                %14 : Tensor = aten::relu(%11)
                %15 : Tensor = aten::add(%13, %14, %9)
                %16 : Tensor = aten::max_pool2d(%15, %6, %6, %5, %8, %7)
                return (%16))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  torch_tensorrt::core::partitioning::PartitionInfo partition_info;
  partition_info.enabled = true;
  partition_info.min_block_size = 3;
  std::unordered_map<torch::jit::Node*, int> fallback_nodes;
  std::vector<torch_tensorrt::core::partitioning::SegmentedBlock> segmented_blocks =
      torch_tensorrt::core::partitioning::segment_graph(g->block(), partition_info, fallback_nodes);
  ASSERT_TRUE(
      checkSegmentedBlockNumber(segmented_blocks, torch_tensorrt::core::partitioning::SegmentedBlock::kTensorRT, 1));
  ASSERT_TRUE(
      checkSegmentedBlockNumber(segmented_blocks, torch_tensorrt::core::partitioning::SegmentedBlock::kTorch, 1));
  ASSERT_TRUE(checkSegmentedBlockNodesMapping(segmented_blocks, g, {{0, 1, 2}, {3, 4, 5, 6}}));
}

TEST(Partitioning, SegmentBranchModelWithForcedFallbackOPCorrectly) {
  const auto graph = R"IR(
                graph(%0 : Tensor,
                      %1 : Float(32, 3, 3, 3, strides=[27, 9, 3, 1]),
                      %2 : Float(32),
                      %3 : Float(16, 32, 3, 3, strides=[288, 9, 3, 1]),
                      %4 : Float(16)):
                  %5 : int[] = prim::Constant[value=[0, 0]]()
                  %6 : int[] = prim::Constant[value=[2, 2]]()
                  %7 : bool = prim::Constant[value=0]()
                  %8 : int[] = prim::Constant[value=[1, 1]]()
                  %9 : int = prim::Constant[value=1]()
                  %10: Tensor = aten::_convolution(%0, %1, %2, %8, %8, %8, %7, %5, %9, %7, %7, %7, %7)
                  %11 : Tensor = aten::_convolution(%10, %3, %4, %8, %8, %8, %7, %5, %9, %7, %7, %7, %7)

                  %12: Tensor = aten::log_sigmoid(%10)

                  %13 : Tensor = aten::_convolution(%12, %3, %4,  %8, %8, %8, %7, %5, %9, %7, %7, %7, %7)

                  %14 : Tensor = aten::relu(%11)

                  %15 : Tensor = aten::add(%13, %14, %9)
                  %16 : Tensor = aten::max_pool2d(%15, %6, %6, %5, %8, %7)
                  return (%16))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  torch_tensorrt::core::partitioning::PartitionInfo partition_info;
  partition_info.enabled = true;
  partition_info.forced_fallback_operators.push_back("aten::relu");
  std::unordered_map<torch::jit::Node*, int> fallback_nodes;
  torch_tensorrt::core::partitioning::PartitionedGraph segmented_blocks =
      torch_tensorrt::core::partitioning::segment_graph(g->block(), partition_info, fallback_nodes);
  ASSERT_TRUE(
      checkSegmentedBlockNumber(segmented_blocks, torch_tensorrt::core::partitioning::SegmentedBlock::kTensorRT, 3));
  ASSERT_TRUE(
      checkSegmentedBlockNumber(segmented_blocks, torch_tensorrt::core::partitioning::SegmentedBlock::kTorch, 2));
  ASSERT_TRUE(checkSegmentedBlockNodesMapping(segmented_blocks, g, {{0, 1}, {2}, {3}, {4}, {5, 6}}));
}
