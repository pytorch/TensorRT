#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/ir/subgraph_matcher.h"

TEST(LoweringPasses, ReduceGeluCorrectly) {
  std::string source_graph = R"IR(
    graph(%x):
        %out : Tensor = aten::gelu(%x)
        return (%out))IR";
  std::string target_graph = R"IR(
    graph(%x.1 : Tensor):
        %6 : float = prim::Constant[value=0.044714999999999998]()
        %5 : float = prim::Constant[value=0.79788456080000003]()
        %4 : float = prim::Constant[value=1.]()
        %3 : float = prim::Constant[value=0.5]()
        %2 : int = prim::Constant[value=1]()
        %7 : Tensor = aten::mul(%x.1, %3)
        %8 : Tensor = aten::mul(%x.1, %5)
        %9 : Tensor = aten::mul(%x.1, %6)
        %10 : Tensor = aten::mul(%9, %x.1)
        %11 : Tensor = aten::add(%10, %4, %2)
        %12 : Tensor = aten::mul(%8, %11)
        %13 : Tensor = aten::tanh(%12)
        %14 : Tensor = aten::add(%13, %4, %2)
        %15 : Tensor = aten::mul(%7, %14)
        return (%15))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);
  torch_tensorrt::core::lowering::passes::ReduceGelu(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}

TEST(LoweringPasses, ReduceGeluApproximateCorrectly) {
  std::string source_graph = R"IR(
    graph(%x, %approx):
        %out : Tensor = aten::gelu(%x, %approx)
        return (%out))IR";
  std::string target_graph = R"IR(
    graph(%x.1 : Tensor, %approx):
        %6 : float = prim::Constant[value=0.044714999999999998]()
        %5 : float = prim::Constant[value=0.79788456080000003]()
        %4 : float = prim::Constant[value=1.]()
        %3 : float = prim::Constant[value=0.5]()
        %2 : int = prim::Constant[value=1]()
        %7 : Tensor = aten::mul(%x.1, %3)
        %8 : Tensor = aten::mul(%x.1, %5)
        %9 : Tensor = aten::mul(%x.1, %6)
        %10 : Tensor = aten::mul(%9, %x.1)
        %11 : Tensor = aten::add(%10, %4, %2)
        %12 : Tensor = aten::mul(%8, %11)
        %13 : Tensor = aten::tanh(%12)
        %14 : Tensor = aten::add(%13, %4, %2)
        %15 : Tensor = aten::mul(%7, %14)
        return (%15))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);
  torch_tensorrt::core::lowering::passes::ReduceGelu(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}
