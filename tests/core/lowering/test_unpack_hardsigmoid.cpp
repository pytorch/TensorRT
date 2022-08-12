#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/ir/subgraph_matcher.h"

TEST(LoweringPasses, UnpackHardSigmoid) {
  std::string source_graph = R"IR(
        graph(%input):
            %result = aten::hardsigmoid(%input)
            return (%result))IR";

  std::string target_graph = R"IR(
        graph(%x.1):
            %22 : float = prim::Constant[value=0.5]()
            %3 : int = prim::Constant[value=6]()
            %5 : int = prim::Constant[value=1]()
            %10 : int = prim::Constant[value=0]()
            %4 : Tensor = aten::div(%x.1, %3)
            %9 : Tensor = aten::add(%4, %22, %5)
            %21 : Tensor = aten::clamp(%9, %10, %5)
            return (%21))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);

  auto in = at::rand({10, 100}, {at::kCUDA});
  auto sg_params = torch_tensorrt::core::ir::get_static_params(sg->inputs(), {});
  auto sg_results = torch_tensorrt::tests::util::RunGraph(sg, sg_params, {in});

  torch_tensorrt::core::lowering::passes::UnpackHardSigmoid(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());

  in = at::clone(in);
  auto tg_params = torch_tensorrt::core::ir::get_static_params(tg->inputs(), {});
  auto tg_results = torch_tensorrt::tests::util::RunGraph(tg, tg_params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(sg_results[0], tg_results[0], 2e-6));
}

TEST(LoweringPasses, UnpackHardSigmoidInPlace) {
  std::string source_graph = R"IR(
        graph(%input):
            %result = aten::hardsigmoid_(%input)
            return (%result))IR";

  std::string target_graph = R"IR(
        graph(%x.1):
            %22 : float = prim::Constant[value=0.5]()
            %3 : int = prim::Constant[value=6]()
            %5 : int = prim::Constant[value=1]()
            %10 : int = prim::Constant[value=0]()
            %4 : Tensor = aten::div(%x.1, %3)
            %9 : Tensor = aten::add(%4, %22, %5)
            %21 : Tensor = aten::clamp(%9, %10, %5)
            return (%21))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);

  auto in = at::rand({10, 100}, {at::kCUDA});
  auto sg_params = torch_tensorrt::core::ir::get_static_params(sg->inputs(), {});
  auto sg_results = torch_tensorrt::tests::util::RunGraph(sg, sg_params, {in});

  torch_tensorrt::core::lowering::passes::UnpackHardSigmoid(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());

  in = at::clone(in);
  auto tg_params = torch_tensorrt::core::ir::get_static_params(tg->inputs(), {});
  auto tg_results = torch_tensorrt::tests::util::RunGraph(tg, tg_params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(sg_results[0], tg_results[0], 2e-6));
}
