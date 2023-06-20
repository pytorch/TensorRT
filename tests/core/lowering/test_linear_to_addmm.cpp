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

TEST(LoweringPasses, LinearToAddMMBiasNoneGraphRun) {
  std::string source_graph = R"IR(
    graph(%input, %weight):
      %biasNone : None = prim::Constant()
      %true : bool = prim::Constant[value=1]()
      %invalid_weight : Tensor = aten::t(%weight)
      %4 : Tensor = prim::If(%true)
        block0():
          %res = aten::linear(%input, %weight, %biasNone)
          -> (%res)
        block1():
          %res = aten::linear(%input, %invalid_weight, %biasNone)
          -> (%res)
      return (%4))IR";

  // This regression test case ensures the Linear-to-AddMM lowering pass satisfies two constraints for non-Tensor bias:
  // 1. It recursively resolves sub-blocks within the node, replacing sub-blocks to be converted as well
  // 2. It does not pre-evaluate branches of the block which may have invalid operations

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, g.get());

  auto in_0 = at::rand({8, 7}, {at::kCUDA});
  auto in_1 = at::rand({8, 7}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in_0, in_1});

  torch_tensorrt::core::lowering::passes::LinearToAddMM(g);

  LOG_DEBUG(g);

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in_0, in_1});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(LoweringPasses, LinearToAddMMBiasGraphRun) {
  std::string source_graph = R"IR(
    graph(%input, %weight, %bias):
      %true : bool = prim::Constant[value=1]()
      %invalid_weight : Tensor = aten::t(%weight)
      %4 : Tensor = prim::If(%true)
        block0():
          %res = aten::linear(%input, %weight, %bias)
          -> (%res)
        block1():
          %res = aten::linear(%input, %invalid_weight, %bias)
          -> (%res)
      return (%4))IR";

  // This regression test case ensures the Linear-to-AddMM lowering pass satisfies two constraints for Tensor bias:
  // 1. It recursively resolves sub-blocks within the node, replacing sub-blocks to be converted as well
  // 2. It does not pre-evaluate branches of the block which may have invalid operations

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, g.get());

  auto in_0 = at::rand({8, 7}, {at::kCUDA});
  auto in_1 = at::rand({8, 7}, {at::kCUDA});
  auto in_2 = at::rand({8, 8}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in_0, in_1, in_2});

  torch_tensorrt::core::lowering::passes::LinearToAddMM(g);

  LOG_DEBUG(g);

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in_0, in_1, in_2});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}
