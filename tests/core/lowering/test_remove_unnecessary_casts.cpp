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
  torch_tensorrt::core::lowering::passes::RemoveContiguous(sg);

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
  torch_tensorrt::core::lowering::passes::RemoveContiguous(sg);

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
  torch_tensorrt::core::lowering::passes::RemoveContiguous(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, tg.get());

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}