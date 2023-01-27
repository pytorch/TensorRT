#include <string>
#include "core/partitioning/partitioning.h"
#include "core/util/trt_util.h"
#include "gtest/gtest.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/script.h"

bool checkInsertedCastNodeNumber(torch_tensorrt::core::partitioning::SegmentedBlock& seg_block, int target_count) {
  int64_t cnt = 0;
  for (auto node : seg_block.nodes()) {
    if (node->kind().toQualString() == std::string("aten::to")) {
      cnt++;
    }
  }
  std::cout << "Found count of " << cnt << " inserted aten::to nodes, (looking for " << target_count
            << " aten::to nodes)" << std::endl;

  return target_count == cnt;
}

TEST(Partitioning, ExplicitNodeAutoConversionCorrectly) {
  const auto graph = R"IR(
          graph(%0 : Tensor,
                %1 : Tensor):
            %2 : int = prim::Constant[value=4]()
            %3 : bool = prim::Constant[value=0]()
            %4 : NoneType = prim::Constant()
            %5 : int = prim::Constant[value=1]()
            %7: Tensor = aten::to(%1, %2, %3, %3, %4)
            %8 : Tensor = aten::mul(%0, %0)
            %9 : Tensor = aten::scatter(%8, %5, %7, %5)
            %10 : Tensor = aten::scatter(%7, %5, %7, %5)
            %12 : Tensor = aten::add(%10, %10, %5)
            return (%9, %12))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get(), true);

  torch_tensorrt::core::partitioning::PartitioningInfo partitioning_info;
  partitioning_info.enabled = true;
  partitioning_info.forced_fallback_operators = {"aten::scatter"};
  partitioning_info.truncate_long_and_double = true;
  std::vector<torch_tensorrt::core::ir::Input> inputs;
  inputs.push_back(torch_tensorrt::core::ir::Input({5, 5}));
  inputs.push_back(torch_tensorrt::core::ir::Input({5, 5}));

  std::unordered_map<const torch::jit::Value*, std::vector<torch_tensorrt::core::ir::Input>> inputs_map;
  std::unordered_map<const torch::jit::Value*, std::vector<c10::optional<at::ScalarType>>> input_types;
  inputs_map.insert({g->inputs()[0], {inputs[0]}});
  input_types.insert({g->inputs()[0], {{at::kFloat}}});
  inputs_map.insert({g->inputs()[1], {inputs[1]}});
  input_types.insert({g->inputs()[1], {{at::kInt}}});

  partitioning_info.collection_input_spec_map = inputs_map;
  torch_tensorrt::core::partitioning::PartitioningCtx ctx(g->block(), partitioning_info);
  ctx.input_types_map = input_types;
  torch_tensorrt::core::partitioning::populateInputIValues(&ctx);
  torch_tensorrt::core::partitioning::partition(&ctx);
  auto segmented_blocks = ctx.partitioned_blocks.begin()->second;

  for (auto& seg_block : segmented_blocks) {
    LOG_DEBUG(seg_block << " cur seg block");
  }
  ASSERT_TRUE(checkInsertedCastNodeNumber(segmented_blocks[1], 2));
}

TEST(Partitioning, ImplicitAutoConversionCorrectly) {
  const auto graph = R"IR(
          graph(%0 : Tensor):
            %2 : int = prim::Constant[value=0]()
            %4 : int = aten::size(%0, %2)
            %6 : Tensor = prim::NumToTensor(%4)
            %2 : int = prim::Constant[value=5]()
            %7 : int[] = prim::ListConstruct(%2, %2)
            %8 : bool = prim::Constant[value=0]()
            %9 : Tensor = aten::expand(%6, %7, %8)

            %10 : Tensor = aten::mul(%9, %9)
            return (%10))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get(), true);

  torch_tensorrt::core::partitioning::PartitioningInfo partitioning_info;
  partitioning_info.enabled = true;
  partitioning_info.forced_fallback_operators = {"aten::expand"};
  partitioning_info.truncate_long_and_double = true;
  std::vector<torch_tensorrt::core::ir::Input> inputs;

  inputs.push_back(torch_tensorrt::core::ir::Input({5, 5}));

  std::unordered_map<const torch::jit::Value*, std::vector<torch_tensorrt::core::ir::Input>> inputs_map;
  std::unordered_map<const torch::jit::Value*, std::vector<c10::optional<at::ScalarType>>> input_types;
  inputs_map.insert({g->inputs()[0], {inputs[0]}});
  input_types.insert({g->inputs()[0], {{at::kFloat}}});

  partitioning_info.collection_input_spec_map = inputs_map;
  torch_tensorrt::core::partitioning::PartitioningCtx ctx(g->block(), partitioning_info);
  ctx.input_types_map = input_types;

  torch_tensorrt::core::partitioning::populateInputIValues(&ctx);
  torch_tensorrt::core::partitioning::partition(&ctx);
  auto segmented_blocks = ctx.partitioned_blocks.begin()->second;

  for (auto& seg_block : segmented_blocks) {
    LOG_DEBUG(seg_block << " cur seg block");
  }
  ASSERT_TRUE(checkInsertedCastNodeNumber(segmented_blocks[1], 2));
}

TEST(Partitioning, ExplicitNodeAutoInt8ConversionCorrectly) {
  const auto graph = R"IR(
          graph(%x.1 : Tensor,
                %y.1 : Tensor):

            %26 : int = prim::Constant[value=1]()
            %21 : bool = prim::Constant[value=0]()
            %60 : Device = prim::Constant[value="cuda"]()
            %14 : NoneType = prim::Constant()
            %3 : int = prim::Constant[value=5]()
            %19 : int = prim::Constant[value=0]()
            %29 : int = prim::Constant[value=2]()
            %13 : int[] = prim::ListConstruct(%3, %3)
            %k_.1 : Tensor = aten::ones(%13, %19, %14, %60, %14)
            %20 : int[] = prim::ListConstruct(%19)
            %k.1 : Tensor = aten::sum(%k_.1, %20, %21, %14)
            %x.5 : Tensor = aten::add_(%x.1, %y.1, %26)
            %31 : Tensor = aten::mul(%y.1, %29)
            %x.9 : Tensor = aten::add_(%x.5, %31, %26)
            %x.13 : Tensor = aten::add_(%x.9, %k.1, %26)
            %x.17 : Tensor = aten::sub_(%x.13, %k.1, %26)
            %x.21 : Tensor = aten::add_(%x.17, %k.1, %26)
            %x.25 : Tensor = aten::sub_(%x.21, %k.1, %26)

            return (%x.25))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get(), true);

  torch_tensorrt::core::partitioning::PartitioningInfo partitioning_info;
  partitioning_info.enabled = true;
  partitioning_info.cast_int8_inputs = true;
  partitioning_info.forced_fallback_operators = {"aten::ones"};
  partitioning_info.truncate_long_and_double = true;
  std::vector<torch_tensorrt::core::ir::Input> inputs;
  inputs.push_back(torch_tensorrt::core::ir::Input({5, 5}));
  inputs.push_back(torch_tensorrt::core::ir::Input({5, 5}));

  std::unordered_map<const torch::jit::Value*, std::vector<torch_tensorrt::core::ir::Input>> inputs_map;
  std::unordered_map<const torch::jit::Value*, std::vector<c10::optional<at::ScalarType>>> input_types;
  inputs_map.insert({g->inputs()[0], {inputs[0]}});
  input_types.insert({g->inputs()[0], {{at::kFloat}}});
  inputs_map.insert({g->inputs()[1], {inputs[1]}});
  input_types.insert({g->inputs()[1], {{at::kInt}}});

  partitioning_info.collection_input_spec_map = inputs_map;
  torch_tensorrt::core::partitioning::PartitioningCtx ctx(g->block(), partitioning_info);
  ctx.input_types_map = input_types;
  torch_tensorrt::core::partitioning::populateInputIValues(&ctx);
  torch_tensorrt::core::partitioning::partition(&ctx);
  auto segmented_blocks = ctx.partitioned_blocks.begin()->second;

  for (auto& seg_block : segmented_blocks) {
    LOG_DEBUG(seg_block << " cur seg block");
  }

  // Seeking 1 inserted aten::to converting Byte to Int (%k_.1 is a Byte Tensor)
  ASSERT_TRUE(checkInsertedCastNodeNumber(segmented_blocks[0], 1));
}
