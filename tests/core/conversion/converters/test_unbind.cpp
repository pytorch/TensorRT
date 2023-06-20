#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ATenUnbindConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
        %2 : int = prim::Constant[value=0]()
        %3 : Tensor[] = aten::unbind(%x.1, %2)
        %o1.1 : Tensor, %o2.1 : Tensor = prim::ListUnpack(%3)
        return (%o1.1, %o2.1))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {2, 3, 4, 4}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  for (size_t i = 0; i < jit_results.size(); i++) {
    auto trt = trt_results[i];
    ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[i], trt, 2e-6));
  }
}

TEST(Converters, ATenUnbindNegativeAxisConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
        %2 : int = prim::Constant[value=-1]()
        %3 : Tensor[] = aten::unbind(%x.1, %2)
        %o1.1 : Tensor, %o2.1 : Tensor = prim::ListUnpack(%3)
        return (%o1.1, %o2.1))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {5, 2}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  for (size_t i = 0; i < jit_results.size(); i++) {
    auto trt = trt_results[i];
    ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[i], trt, 2e-6));
  }
}

TEST(Converters, ATenUnbindEvaluatedTensor) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %2 : None = prim::Constant()
        %3 : int[] = aten::size(%x.1)
        %z.1 : Tensor = aten::zeros(%3, %2, %2, %2, %2)
        %5 : int = prim::Constant[value=-1]()
        %6 : Tensor[] = aten::unbind(%z.1, %5)
        %o1.1 : Tensor, %o2.1 : Tensor = prim::ListUnpack(%6)
        return (%o1.1, %o2.1))IR";

  auto in = at::randint(1, 10, {2}, {at::kCUDA});

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  for (size_t i = 0; i < jit_results.size(); i++) {
    auto trt = trt_results[i];
    ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[i].cuda(), trt, 2e-6));
  }
}