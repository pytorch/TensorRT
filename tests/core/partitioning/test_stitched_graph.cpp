#include <string>
#include <unordered_set>
#include "core/compiler.h"
#include "core/util/trt_util.h"
#include "gtest/gtest.h"
#include "torch/csrc/jit/ir/constants.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/script.h"

bool checkAllInputsExistInStitchedGraph(std::shared_ptr<torch::jit::Graph> g) {
  std::unordered_set<torch::jit::Value*> available_values;
  for (auto v : g->inputs()) {
    available_values.insert(v);
  }
  for (const auto n : g->nodes()) {
    for (auto input : n->inputs()) {
      if (!available_values.count(input))
        return false;
    }
    for (auto output : n->outputs()) {
      available_values.insert(output);
    }
  }
  return true;
}

TEST(Partitioning, StitchSequentialModelSegmentedBlockCorrectly) {
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

  auto parsed_g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, parsed_g.get());

  auto g = std::make_shared<torch::jit::Graph>();
  std::vector<std::vector<int64_t>> all_shapes{{32, 3, 3, 3}, {32}, {16, 32, 3, 3}, {16}, {8, 16, 3, 3}, {8}};
  std::unordered_map<torch::jit::Value*, torch::jit::Value*> tensor_to_constant;
  for (size_t i = 0; i < all_shapes.size(); ++i) {
    auto in = at::randint(5, all_shapes[i], {at::kCUDA});
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

  torch::jit::script::Module mod(c10::QualifiedName("module"));

  auto self = g->insertInput(0, "self_1");
  self->setType(mod.type());
  auto cur_method = mod._ivalue()->compilation_unit()->create_function(c10::QualifiedName("forward"), g);
  auto schema = torch_tensorrt::core::util::GenerateGraphSchema(cur_method->name(), g);
  mod.type()->addMethod(cur_method);
  cur_method->setSchema(schema);

  std::vector<torch_tensorrt::core::ir::Input> inputs;
  inputs.push_back(torch_tensorrt::core::ir::Input({3, 3, 16, 16}));
  torch_tensorrt::core::CompileSpec cfg(inputs);
  cfg.partition_info.enabled = true;
  torch::jit::script::Module new_mod = torch_tensorrt::core::CompileGraph(mod, cfg);
  auto fallback_g = new_mod.get_method("forward").graph();
  ASSERT_TRUE(checkAllInputsExistInStitchedGraph(fallback_g));
}

TEST(Partitioning, StitchBranchModelSegmentedBlockCorrectly) {
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

  auto parsed_g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, parsed_g.get());

  auto g = std::make_shared<torch::jit::Graph>();
  std::vector<std::vector<int64_t>> all_shapes{{32, 3, 3, 3}, {32}, {16, 32, 3, 3}, {16}};
  std::unordered_map<torch::jit::Value*, torch::jit::Value*> tensor_to_constant;
  for (size_t i = 0; i < all_shapes.size(); ++i) {
    auto in = at::randint(5, all_shapes[i], {at::kCUDA});
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

  torch::jit::script::Module mod(c10::QualifiedName("module"));

  auto self = g->insertInput(0, "self_1");
  self->setType(mod.type());
  auto cur_method = mod._ivalue()->compilation_unit()->create_function(c10::QualifiedName("forward"), g);
  auto schema = torch_tensorrt::core::util::GenerateGraphSchema(cur_method->name(), g);
  mod.type()->addMethod(cur_method);
  cur_method->setSchema(schema);

  std::vector<torch_tensorrt::core::ir::Input> inputs;
  inputs.push_back(torch_tensorrt::core::ir::Input({3, 3, 16, 16}));
  torch_tensorrt::core::CompileSpec cfg(inputs);
  cfg.partition_info.enabled = true;
  torch::jit::script::Module new_mod = torch_tensorrt::core::CompileGraph(mod, cfg);
  auto fallback_g = new_mod.get_method("forward").graph();
  ASSERT_TRUE(checkAllInputsExistInStitchedGraph(fallback_g));
}
