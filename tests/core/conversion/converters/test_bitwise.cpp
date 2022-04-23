#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ATenBitwiseNotConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %3 : Tensor = aten::bitwise_not(%0)
        return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(-128, 128, {10}, {at::kCUDA});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs, {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  
  ASSERT_TRUE(torch_tensorrt::tests::util::exactlyEqual(jit_results[0], trt_results[0]);
}
