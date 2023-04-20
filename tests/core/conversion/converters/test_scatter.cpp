#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ScatterValueConvertsCorrectly) {
  const auto graph = R"IR(
        graph(%data : Tensor,
              %index.1 : Tensor):
          %value : int = prim::Constant[value=100]()
          %dim : int = prim::Constant[value=1]()
          %5 : NoneType = prim::Constant()
          %6 : bool = prim::Constant[value=0]()
          %7 : int = prim::Constant[value=4]()
          %index : Tensor = aten::to(%index.1, %7, %6, %6, %5)
          %10 : Tensor = aten::scatter(%data, %dim, %index, %value)
          return (%10))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto index = at::randint(0, 5, {2, 2}, {at::kCUDA});
  auto data = at::randn({5, 5}, {at::kCUDA});

  auto jit_index = at::clone(index);
  auto jit_data = at::clone(data);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_data, jit_index});

  auto trt_index = at::clone(index);
  auto trt_data = at::clone(data);
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_data, trt_index});

  for (size_t i = 0; i < jit_results.size(); i++) {
    auto trt = trt_results[i].reshape(jit_results[i].sizes());
    ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[i], trt, 2e-6));
  }
}

TEST(Converters, ScatterSrcConvertsCorrectly) {
  const auto graph = R"IR(
        graph(%data : Tensor,
              %src : Tensor,
              %index.1 : Tensor):
          %dim : int = prim::Constant[value=1]()
          %5 : NoneType = prim::Constant()
          %6 : bool = prim::Constant[value=0]()
          %7 : int = prim::Constant[value=4]()
          %index : Tensor = aten::to(%index.1, %7, %6, %6, %5)
          %10 : Tensor = aten::scatter(%data, %dim, %index, %src)
          return (%10))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto index = at::randint(0, 4, {2, 2}, {at::kCUDA});
  auto data = at::randn({5, 5}, {at::kCUDA});
  auto src = at::randn({2, 2}, {at::kCUDA});

  auto jit_index = at::clone(index);
  auto jit_data = at::clone(data);
  auto jit_src = at::clone(src);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_data, jit_src, jit_index});

  auto trt_index = at::clone(index);
  auto trt_data = at::clone(data);
  auto trt_src = at::clone(src);
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_data, trt_src, trt_index});

  for (size_t i = 0; i < jit_results.size(); i++) {
    auto trt = trt_results[i].reshape(jit_results[i].sizes());
    ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[i], trt, 2e-6));
  }
}