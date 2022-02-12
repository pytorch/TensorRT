#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/ir/subgraph_matcher.h"

TEST(LoweringPasses, RemoveUnnecessaryCastIntCorrectly) {
  std::string source_graph = R"IR(
    graph(%1: int):
      %2: Tensor = aten::NumToTensor(%1)
      %3: int = aten::Int(%2)
      %4: int = aten::add(%3, %3, %3)
      return (%4))IR";
  std::string target_graph = R"IR(
    graph(%1: int):
      %4: int = aten::add(%1, %1, %1)
      return (%4))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, sg.get());
  torch_tensorrt::core::lowering::passes::RemoveUnnecessaryCasts(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, tg.get());

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}

TEST(LoweringPasses, RemoveUnnecessaryCastFloatCorrectly) {
  std::string source_graph = R"IR(
    graph(%1: float):
      %2: Tensor = aten::NumToTensor(%1)
      %3: float = aten::Float(%2)
      %4: float = aten::add(%3, %3, %3)
      return (%3))IR";
  std::string target_graph = R"IR(
    graph(%1: float):
      %4: float = aten::add(%1, %1, %1)
      return (%4))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, sg.get());
  torch_tensorrt::core::lowering::passes::RemoveUnnecessaryCasts(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, tg.get());

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}

TEST(LoweringPasses, RemoveUnnecessaryCastBoolCorrectly) {
  std::string source_graph = R"IR(
    graph(%1: bool):
      %2: Tensor = aten::NumToTensor(%1)
      %3: bool = aten::Bool(%2)
      %4: bool = aten::__and__(%3, %3)
      return (%3))IR";
  std::string target_graph = R"IR(
    graph(%1: bool):
      %4: bool = aten::__and__(%1, %1)
      return (%4))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, sg.get());
  torch_tensorrt::core::lowering::passes::RemoveUnnecessaryCasts(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, tg.get());

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}

TEST(LoweringPasses, RemoveSingleUse0DTensorsIntCorrectly) {
  std::string source_graph = R"IR(
    graph(%0: int):
      %1: Tensor = prim::Constant[value=[8]]()
      %2: int = prim::Constant[value=1]()
      %3: Tensor = prim::NumToTensor(%0)
      %4: Tensor = aten::add(%1, %3, %2)
      %5: int = aten::Int(%4)
      %6: int = aten::add(%5, %5)
      return (%6))IR";
  std::string target_graph = R"IR(
    graph(%0: int):
      %1: int = prim::Constant[value=8]()
      %4: int = aten::add(%1, %0)
      %6: int = aten::add(%4, %4)
      return (%6))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, sg.get());

  auto first_op = *(sg->block()->nodes().begin());
  torch::jit::WithInsertPoint guard(first_op);
  torch::jit::Value* r = sg->insertConstant(
      c10::scalar_to_tensor(8), c10::nullopt, first_op->scope());
  r->copyMetadata(first_op->output());
  r->setType(c10::TensorType::get());
  first_op->output()->replaceAllUsesWith(r);
  first_op->destroy();

  torch_tensorrt::core::lowering::passes::RemoveSingleUse0DTensors(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, tg.get());

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}

TEST(LoweringPasses, RemoveSingleUse0DTensorsFloatCorrectly) {
  std::string source_graph = R"IR(
    graph(%0: float):
      %1: Tensor = prim::Constant[value=[8.]]()
      %2: float = prim::Constant[value=1.]()
      %3: Tensor = prim::NumToTensor(%0)
      %4: Tensor = aten::add(%1, %3, %2)
      %5: float = aten::Float(%4)
      %6: float = aten::add(%5, %5)
      return (%6))IR";
  std::string target_graph = R"IR(
    graph(%0: float):
      %1: float = prim::Constant[value=8.]()
      %4: float = aten::add(%1, %0)
      %6: float = aten::add(%4, %4)
      return (%6))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, sg.get());

  auto first_op = *(sg->block()->nodes().begin());
  torch::jit::WithInsertPoint guard(first_op);
  torch::jit::Value* r = sg->insertConstant(
      c10::scalar_to_tensor(8.0), c10::nullopt, first_op->scope());
  r->copyMetadata(first_op->output());
  r->setType(c10::TensorType::get());
  first_op->output()->replaceAllUsesWith(r);
  first_op->destroy();

  torch_tensorrt::core::lowering::passes::RemoveSingleUse0DTensors(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, tg.get());

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}