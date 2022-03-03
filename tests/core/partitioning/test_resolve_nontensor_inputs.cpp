#include <string>
#include "core/compiler.h"
#include "core/util/trt_util.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/script.h"

bool checkSegmentedBlockInputType(const torch_tensorrt::core::partitioning::SegmentedBlock &segmented_block,
                                  const std::function<bool(torch::jit::TypePtr)> &condition) {
  for (auto input : segmented_block.raw_inputs()) {
    if (!condition(input->type())) {
      return false;
    }
  }
  return true;
}

int count_trt_engines(std::shared_ptr<torch::jit::Graph> g) {
  int count = 0;
  for (const auto n : g->nodes()) {
    if (n->kind().toQualString() == std::string("tensorrt::execute_engine")) {
      ++count;
    }
  }
  return count;
}

TEST(Partitioning, ResolveNonTensorInputsForIFBlockCorrectly) {
  const auto graph = R"IR(
        graph(%x : Tensor, %y : Tensor):
          %0 : int = prim::Constant[value=0]()
          %1 : int = prim::Constant[value=1]()
          %a : Tensor = aten::add(%x, %y, %1)
          %s : int = aten::size(%a, %1)
          %D3.1 : Tensor = prim::NumToTensor(%s)
          %19 : bool = aten::is_floating_point(%D3.1)
          %2 : Tensor = prim::If(%19)
            block0():
                %2.1 : Tensor = aten::sub(%a, %y, %1)
                -> (%2.1)
            block1():
                %2.2 : Tensor = aten::sub(%a, %y, %0)
                -> (%2.2)
          %3 : Tensor = prim::If(%19)
            block0():
                %3.1 : Tensor = aten::sub(%a, %y, %1)
                -> (%3.1)
            block1():
                %3.2 : Tensor = aten::sub(%a, %y, %0)
                -> (%3.2)
          %4 : Tensor = aten::add(%2, %3, %1)
          return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  std::vector<torch_tensorrt::core::ir::Input> inputs;
  inputs.push_back(torch_tensorrt::core::ir::Input({3, 4}));
  inputs.push_back(torch_tensorrt::core::ir::Input({3, 4}));
  torch_tensorrt::core::CompileSpec cfg(inputs);
  cfg.partition_info.enabled = true;
  cfg.partition_info.forced_fallback_operators.push_back("aten::sub");
  cfg.convert_info.engine_settings.truncate_long_and_double = true;
  cfg.partition_info.truncate_long_and_double = true;

  torch::jit::script::Module mod(c10::QualifiedName("module"));

  auto self = g->insertInput(0, "self_1");
  self->setType(mod.type());
  auto cur_method = mod._ivalue()->compilation_unit()->create_function(c10::QualifiedName("forward"), g);
  auto schema = torch_tensorrt::core::util::GenerateGraphSchema(cur_method->name(), g);
  mod.type()->addMethod(cur_method);
  cur_method->setSchema(schema);

  torch::jit::script::Module new_mod = torch_tensorrt::core::CompileGraph(mod, cfg);

  auto in0 = at::randint(5, {3, 4}, {at::kCUDA});
  auto in1 = at::randint(5, {3, 4}, {at::kCUDA});

  auto jit_in0 = at::clone(in0);
  auto jit_in1 = at::clone(in1);
  auto trt_in0 = at::clone(in0);
  auto trt_in1 = at::clone(in1);

  auto jit_results = mod.forward({jit_in0, jit_in1});
  auto trt_results = new_mod.forward({trt_in0, trt_in1});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results.toTensor(), trt_results.toTensor(), 2e-6));
}

TEST(Partitioning, ResolveNonTensorInputsCorrectly) {
  const auto graph = R"IR(
          graph(%0 : Float(1, 3, 16, 16, strides=[768, 256, 16, 1]),
                %1 : Float(16, 3, 3, 3, strides=[27, 9, 3, 1]),
                %2 : Float(16, strides=[1])):
            %3 : int[] = prim::Constant[value=[0, 0]]()
            %4 : int[] = prim::Constant[value=[1, 1]]()
            %5 : bool = prim::Constant[value=0]()
            %6 : bool = prim::Constant[value=1]()
            %7 : int = prim::Constant[value=0]()
            %8 : int = aten::size(%0, %7)
            %9 : Tensor = aten::log_sigmoid(%0)
            %10 : Tensor = aten::_convolution(%9, %1, %2, %4, %3, %4, %5, %3, %8, %5, %5, %6, %6)
            %11 : Tensor = aten::relu(%10)
            return (%11))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  torch_tensorrt::core::partitioning::PartitionInfo partition_info;
  partition_info.enabled = true;
  std::vector<torch_tensorrt::core::ir::Input> inputs;
  inputs.push_back(torch_tensorrt::core::ir::Input({1, 3, 16, 16}));
  inputs.push_back(torch_tensorrt::core::ir::Input({16, 3, 3, 3}));
  inputs.push_back(torch_tensorrt::core::ir::Input({16}));

  std::unordered_map<const torch::jit::Value*, torch_tensorrt::core::ir::Input> inputs_map;
  std::unordered_map<const torch::jit::Value*, c10::optional<at::ScalarType>> input_types;
  for (size_t i = 0; i < g->inputs().size(); ++i) {
    inputs_map.insert({g->inputs()[i], inputs[i]});
    input_types.insert({g->inputs()[i], {at::kFloat}});
  }
  auto input_ivalues_map = torch_tensorrt::core::partitioning::generateRandomInputs(inputs_map, input_types);
  std::vector<torch_tensorrt::core::partitioning::SegmentedBlock> segmented_blocks =
      torch_tensorrt::core::partitioning::Partition(g->block(), input_ivalues_map, partition_info);

  int torch_block_cnt = 0, trt_block_cnt = 0;
  for (const auto &segmented_block: segmented_blocks) {
    if (segmented_block.target() == torch_tensorrt::core::partitioning::SegmentedBlock::kTensorRT) {
      ++trt_block_cnt;
      ASSERT_TRUE(checkSegmentedBlockInputType(
            segmented_block,
            [] (torch::jit::TypePtr type_ptr) { return type_ptr->isSubtypeOf(torch::jit::TensorType::get()); }));
    } else {
      ++torch_block_cnt;
      ASSERT_TRUE(checkSegmentedBlockInputType(
            segmented_block,
            [] (torch::jit::TypePtr type_ptr) {
              return type_ptr->isSubtypeOf(torch::jit::TensorType::get()) ||
                  type_ptr->isSubtypeOf(torch::jit::ListType::ofTensors());
            }));
    }
  }
  ASSERT_TRUE(trt_block_cnt == 1 && torch_block_cnt == 1);
}

TEST(Partitioning, ResolveTensorListInputsInTrtCorrectly) {
  const auto graph = R"IR(
          graph(%0 : Float(1, 3, 16, 16, strides=[768, 256, 16, 1]),
                %1 : Float(16, 6, 3, 3, strides=[54, 9, 3, 1]),
                %2 : Float(16, strides=[1])):
            %3 : int[] = prim::Constant[value=[0, 0]]()
            %4 : int[] = prim::Constant[value=[1, 1]]()
            %5 : bool = prim::Constant[value=0]()
            %6 : bool = prim::Constant[value=1]()
            %7 : int = prim::Constant[value=1]()
            %8 : int = prim::Constant[value=0]()
            %9 : Tensor[] = prim::ListConstruct(%0, %0)
            %10 : Tensor = aten::cat(%9, %8)
            %11 : Tensor = aten::log_sigmoid(%10)
            %12 : Tensor = aten::cat(%9, %7)
            %13 : Tensor = aten::_convolution(%12, %1, %2, %4, %3, %4, %5, %3, %7, %5, %5, %6, %6)
            %14 : Tensor = aten::relu(%13)
            %15 : (Tensor, Tensor) = prim::TupleConstruct(%11, %14)
            return (%15))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  torch_tensorrt::core::partitioning::PartitionInfo partition_info;
  partition_info.enabled = true;
  std::vector<torch_tensorrt::core::ir::Input> inputs;
  inputs.push_back(torch_tensorrt::core::ir::Input({1, 3, 16, 16}));
  inputs.push_back(torch_tensorrt::core::ir::Input({16, 6, 3, 3}));
  inputs.push_back(torch_tensorrt::core::ir::Input({16}));

  std::unordered_map<const torch::jit::Value*, torch_tensorrt::core::ir::Input> inputs_map;
  std::unordered_map<const torch::jit::Value*, c10::optional<at::ScalarType>> input_types;
  for (size_t i = 0; i < g->inputs().size(); ++i) {
    inputs_map.insert({g->inputs()[i], inputs[i]});
    input_types.insert({g->inputs()[i], {at::kFloat}});
  }
  auto input_ivalues_map = torch_tensorrt::core::partitioning::generateRandomInputs(inputs_map, input_types);
  std::vector<torch_tensorrt::core::partitioning::SegmentedBlock> segmented_blocks =
      torch_tensorrt::core::partitioning::Partition(g->block(), input_ivalues_map, partition_info);

  int torch_block_cnt = 0, trt_block_cnt = 0;
  for (const auto &segmented_block: segmented_blocks) {
    if (segmented_block.target() == torch_tensorrt::core::partitioning::SegmentedBlock::kTensorRT) {
      ++trt_block_cnt;
      ASSERT_TRUE(checkSegmentedBlockInputType(
            segmented_block,
            [] (torch::jit::TypePtr type_ptr) { return type_ptr->isSubtypeOf(torch::jit::TensorType::get()); }));
    } else {
      ++torch_block_cnt;
      ASSERT_TRUE(checkSegmentedBlockInputType(
            segmented_block,
            [] (torch::jit::TypePtr type_ptr) {
              return type_ptr->isSubtypeOf(torch::jit::TensorType::get()) ||
                  type_ptr->isSubtypeOf(torch::jit::ListType::ofTensors());
            }));
    }
  }
  ASSERT_TRUE(trt_block_cnt == 2 && torch_block_cnt == 2);
}

TEST(Partitioning, ConvertForTensorListInputsInFallbackCorrectly) {
  const auto graph = R"IR(
          graph(%0 : Float(1, 3, 16, 16, strides=[768, 256, 16, 1]),
                %1 : Float(16, 6, 3, 3, strides=[54, 9, 3, 1]),
                %2 : Float(16, strides=[1])):
            %3 : int[] = prim::Constant[value=[0, 0]]()
            %4 : int[] = prim::Constant[value=[1, 1]]()
            %5 : bool = prim::Constant[value=0]()
            %6 : bool = prim::Constant[value=1]()
            %7 : int = prim::Constant[value=1]()
            %8 : int = prim::Constant[value=0]()
            %9 : Tensor[] = prim::ListConstruct(%0, %0)
            %11 : Tensor = aten::log_sigmoid(%0)
            %12 : Tensor = aten::cat(%9, %7)
            %13 : Tensor = aten::_convolution(%12, %1, %2, %4, %3, %4, %5, %3, %7, %5, %5, %6, %6)
            %14 : Tensor = aten::relu(%13)
            %15 : (Tensor, Tensor) = prim::TupleConstruct(%11, %14)
            return (%15))IR";
  auto parsed_g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, parsed_g.get());

  auto g = std::make_shared<torch::jit::Graph>();
  std::vector<std::vector<int64_t>> all_shapes{{16, 6, 3, 3}, {16}};
  std::unordered_map<torch::jit::Value*, torch::jit::Value*> tensor_to_constant;
  for (size_t i = 0; i < all_shapes.size(); ++i) {
    auto in = at::randn(all_shapes[i], {at::kCUDA});
    torch::jit::IValue cur_val = in.clone();
    auto new_val = g->insertConstant(cur_val);
    tensor_to_constant[parsed_g->inputs()[i + 1]] = new_val;
  }
  for (auto node : parsed_g->nodes()) {
    if (node->kind() == torch::jit::prim::Constant)
      continue;
    torch_tensorrt::core::util::cloneNode(node, g, tensor_to_constant);
  }
  g->registerOutput(tensor_to_constant[parsed_g->outputs()[0]]);

  std::vector<torch_tensorrt::core::ir::Input> inputs;
  inputs.push_back(torch_tensorrt::core::ir::Input({1, 3, 16, 16}));
  torch_tensorrt::core::CompileSpec cfg(inputs);
  cfg.partition_info.enabled = true;
  torch::jit::script::Module mod(c10::QualifiedName("module"));

  auto self = g->insertInput(0, "self_1");
  self->setType(mod.type());
  auto cur_method = mod._ivalue()->compilation_unit()->create_function(c10::QualifiedName("forward"), g);
  auto schema = torch_tensorrt::core::util::GenerateGraphSchema(cur_method->name(), g);
  mod.type()->addMethod(cur_method);
  cur_method->setSchema(schema);

  torch::jit::script::Module new_mod = torch_tensorrt::core::CompileGraph(mod, cfg);
  auto fallback_g = new_mod.get_method("forward").graph();
  int count = count_trt_engines(fallback_g);
  ASSERT_TRUE(count == 2);
}
