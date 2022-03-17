#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/ir/subgraph_matcher.h"

TEST(LoweringPasses, LinearToAddMM) {
  std::string source_graph = R"IR(
    graph(%input, %6, %7, %weight, %bias):
      %flat = aten::flatten(%input, %6, %7)
      %res = aten::linear(%flat, %weight, %bias)
      return (%res))IR";
  std::string target_graph = R"IR(
    graph(%input, %6, %7, %weight_t, %bias):
      %1: int = prim::Constant[value=1]()
      %flat = aten::flatten(%input, %6, %7)
      %weight = aten::t(%weight_t)
      %mm: Tensor = aten::matmul(%flat, %weight)
      %b_f: Tensor = trt::const(%bias)
      %out: Tensor = aten::add(%b_f, %mm, %1)
      return (%out))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);
  torch_tensorrt::core::lowering::passes::LinearToAddMM(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}

TEST(LoweringPasses, LinearToAddMMBiasNone) {
  std::string source_graph = R"IR(
    graph(%input, %weight):
      %bias : None = prim::Constant()
      %res = aten::linear(%input, %weight, %bias)
      return (%res))IR";
  std::string target_graph = R"IR(
    graph(%input, %weight_t):
      %weight = aten::t(%weight_t)
      %mm: Tensor = aten::matmul(%input, %weight)
      return (%mm))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);
  torch_tensorrt::core::lowering::passes::LinearToAddMM(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}
