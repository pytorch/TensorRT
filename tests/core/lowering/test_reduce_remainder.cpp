#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/ir/subgraph_matcher.h"

TEST(LoweringPasses, ReduceRemainderCorrectly) {
  std::string source_graph = R"IR(
        graph(%self : Tensor, %other : Tensor):
            %out : Tensor = aten::remainder(%self, %other)
            return (%out))IR";
  std::string target_graph = R"IR(
        graph(%self : Tensor, %other : Tensor):
                %alpha : int = prim::Constant[value=1]()
                %floor: Tensor = aten::floor_divide(%self, %other)
                %prod: Tensor = aten::mul(%floor, %other)
                %out: Tensor = aten::sub(%self, %prod, %alpha)
                return (%out))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);
  torch_tensorrt::core::lowering::passes::ReduceRemainder(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}

TEST(LoweringPasses, ReduceRemainderScalarCorrectly) {
  std::string source_graph = R"IR(
        graph(%self : Tensor, %other : Scalar):
            %out : Tensor = aten::remainder(%self, %other)
            return (%out))IR";
  std::string target_graph = R"IR(
        graph(%self : Tensor, %other : Scalar):
            %alpha : int = prim::Constant[value=1]()
            %floor: Tensor = aten::floor_divide(%self, %other)
            %prod: Tensor = aten::mul(%floor, %other)
            %out: Tensor = aten::sub(%self, %prod, %alpha)
            return (%out))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);
  torch_tensorrt::core::lowering::passes::ReduceRemainder(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}
