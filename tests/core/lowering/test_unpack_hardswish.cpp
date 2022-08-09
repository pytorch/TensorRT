#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/ir/subgraph_matcher.h"

TEST(LoweringPasses, UnpackHardSwish) {
  std::string source_graph = R"IR(
        graph(%input):
            %result = aten::hardswish(%input)
            return (%result))IR";

  std::string target_graph = R"IR(
        graph(%input):
            %1 : Scalar = prim::Constant[value=3.]()
            %2 : Scalar = prim::Constant[value=1.]()
            %3 = aten::add(%input, %1, %2)
            %4 : Scalar = prim::Constant[value=0.]()
            %5 : Scalar = prim::Constant[value=6.]()
            %6 = aten::hardtanh(%3, %4, %5)
            %7 = aten::div(%6, %5)
            %8 = aten::mul(%input, %7)
            return (%8))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);

  auto in = at::rand({10, 100}, {at::kCUDA});
  auto sg_params = torch_tensorrt::core::ir::get_static_params(sg->inputs(), {});
  auto sg_results = torch_tensorrt::tests::util::RunGraph(sg, sg_params, {in});

  torch_tensorrt::core::lowering::passes::UnpackHardSwish(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());

  in = at::clone(in);
  auto tg_params = torch_tensorrt::core::ir::get_static_params(tg->inputs(), {});
  auto tg_results = torch_tensorrt::tests::util::RunGraph(tg, tg_params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(sg_results[0], tg_results[0], 2e-6));
}

TEST(LoweringPasses, UnpackHardInplaceSwish) {
  std::string source_graph = R"IR(
        graph(%input):
            %result = aten::hardswish_(%input)
            return (%result))IR";

  std::string target_graph = R"IR(
        graph(%input):
            %1 : Scalar = prim::Constant[value=3.]()
            %2 : Scalar = prim::Constant[value=1.]()
            %3 = aten::add(%input, %1, %2)
            %4 : Scalar = prim::Constant[value=0.]()
            %5 : Scalar = prim::Constant[value=6.]()
            %6 = aten::hardtanh(%3, %4, %5)
            %7 = aten::div(%6, %5)
            %8 = aten::mul(%input, %7)
            return (%8))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);

  auto in = at::rand({10, 100}, {at::kCUDA});
  auto sg_params = torch_tensorrt::core::ir::get_static_params(sg->inputs(), {});
  auto sg_results = torch_tensorrt::tests::util::RunGraph(sg, sg_params, {in});

  torch_tensorrt::core::lowering::passes::UnpackHardSwish(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());

  in = at::clone(in);
  auto tg_params = torch_tensorrt::core::ir::get_static_params(tg->inputs(), {});
  auto tg_results = torch_tensorrt::tests::util::RunGraph(tg, tg_params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(sg_results[0], tg_results[0], 2e-6));
}
