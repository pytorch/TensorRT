#include <string>
#include "core/partitioning/partitioning.h"
#include "core/util/trt_util.h"
#include "gtest/gtest.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/script.h"

bool checkSegmentedBlockInputShape(
    std::vector<torch_tensorrt::core::partitioning::SegmentedBlock>& segmented_blocks,
    std::vector<std::vector<std::vector<int>>> in_shape) {
  if (segmented_blocks.size() != in_shape.size())
    return false;
  for (size_t i = 0; i < segmented_blocks.size(); ++i) {
    auto cur_block_in_shapes = segmented_blocks[i].in_shapes();
    if (cur_block_in_shapes.size() != in_shape[i].size())
      return false;
    for (size_t j = 0; j < cur_block_in_shapes.size(); ++j) {
      auto cur_input_shape = torch_tensorrt::core::util::toVec(cur_block_in_shapes[j].input_shape);
      for (size_t k = 0; k < cur_input_shape.size(); ++k) {
        if (cur_input_shape[k] != in_shape[i][j][k])
          return false;
      }
    }
  }
  return true;
}

TEST(Partitioning, InferSequentialModelSegmentedBlockShapeCorrectly) {
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
  std::vector<torch_tensorrt::core::ir::Input> inputs;
  inputs.push_back(torch_tensorrt::core::ir::Input({3, 3, 16, 16}));
  inputs.push_back(torch_tensorrt::core::ir::Input({32, 3, 3, 3}));
  inputs.push_back(torch_tensorrt::core::ir::Input({32}));
  inputs.push_back(torch_tensorrt::core::ir::Input({16, 32, 3, 3}));
  inputs.push_back(torch_tensorrt::core::ir::Input({16}));
  inputs.push_back(torch_tensorrt::core::ir::Input({8, 16, 3, 3}));
  inputs.push_back(torch_tensorrt::core::ir::Input({8}));

  std::unordered_map<const torch::jit::Value*, std::vector<torch_tensorrt::core::ir::Input>> inputs_map;
  std::unordered_map<const torch::jit::Value*, std::vector<c10::optional<at::ScalarType>>> input_types;
  for (size_t i = 0; i < g->inputs().size(); ++i) {
    inputs_map.insert({g->inputs()[i], {inputs[i]}});
    input_types.insert({g->inputs()[i], {{at::kFloat}}});
  }
  auto input_ivalues_map = torch_tensorrt::core::partitioning::generateRandomInputs(inputs_map, input_types);
  std::unordered_map<torch::jit::Node*, int> fallback_nodes;
  std::vector<torch_tensorrt::core::partitioning::SegmentedBlock> segmented_blocks =
      torch_tensorrt::core::partitioning::Partition(g->block(), input_ivalues_map, partition_info, fallback_nodes);

  ASSERT_TRUE(checkSegmentedBlockInputShape(
      segmented_blocks,
      {{{3, 3, 16, 16}, {32, 3, 3, 3}, {32}, {16, 32, 3, 3}, {16}},
       {{3, 16, 16, 16}},
       {{3, 16, 16, 16}, {8, 16, 3, 3}, {8}}}));
}

TEST(Partitioning, InferBranchModelSegmentedBlockShapeCorrectly) {
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
  std::vector<torch_tensorrt::core::ir::Input> inputs;
  inputs.push_back(torch_tensorrt::core::ir::Input({3, 3, 16, 16}));
  inputs.push_back(torch_tensorrt::core::ir::Input({32, 3, 3, 3}));
  inputs.push_back(torch_tensorrt::core::ir::Input({32}));
  inputs.push_back(torch_tensorrt::core::ir::Input({16, 32, 3, 3}));
  inputs.push_back(torch_tensorrt::core::ir::Input({16}));

  std::unordered_map<const torch::jit::Value*, std::vector<torch_tensorrt::core::ir::Input>> inputs_map;
  std::unordered_map<const torch::jit::Value*, std::vector<c10::optional<at::ScalarType>>> input_types;
  for (size_t i = 0; i < g->inputs().size(); ++i) {
    inputs_map.insert({g->inputs()[i], {inputs[i]}});
    input_types.insert({g->inputs()[i], {{at::kFloat}}});
  }
  auto input_ivalues_map = torch_tensorrt::core::partitioning::generateRandomInputs(inputs_map, input_types);
  std::unordered_map<torch::jit::Node*, int> fallback_nodes;
  std::vector<torch_tensorrt::core::partitioning::SegmentedBlock> segmented_blocks =
      torch_tensorrt::core::partitioning::Partition(g->block(), input_ivalues_map, partition_info, fallback_nodes);

  ASSERT_TRUE(checkSegmentedBlockInputShape(
      segmented_blocks,
      {{{3, 3, 16, 16}, {32, 3, 3, 3}, {32}, {16, 32, 3, 3}, {16}},
       {{3, 32, 16, 16}},
       {{3, 32, 16, 16}, {16, 32, 3, 3}, {16}, {3, 16, 16, 16}}}));
}
