#include <torch/torch.h>
#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, WhereConvertsCorrectly) {
  const auto graph = R"IR(
        graph(%condition : Tensor,
              %x : Tensor,
              %y : Tensor):
          %out : Tensor = aten::where(%condition, %x, %y)
          return (%out))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto condition = at::randint(0, 2, {5, 5}, {at::kCUDA}).to(torch::kBool);
  auto x = at::randn({5, 5}, {at::kCUDA});
  auto y = at::randn({5, 5}, {at::kCUDA});

  auto jit_condition = at::clone(condition);
  auto jit_x = at::clone(x);
  auto jit_y = at::clone(y);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_condition, jit_x, jit_y});

  auto trt_condition = at::clone(condition);
  auto trt_x = at::clone(x);
  auto trt_y = at::clone(y);
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_condition, trt_x, trt_y});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, WhereConvertsMismatchedShapesCorrectly) {
  const auto graph = R"IR(
        graph(%condition : Tensor,
              %x : Tensor,
              %y : Tensor):
          %out : Tensor = aten::where(%condition, %x, %y)
          return (%out))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  // As per Torch behavior, the input Tensors are expected to be broadcasted
  // along their respective dimension in the largest-rank Tensor provided
  auto condition = at::randint(0, 2, {7, 5}, {at::kCUDA}).to(torch::kBool);
  auto x = at::randn({2, 7, 5}, {at::kCUDA});
  auto y = at::randn({5}, {at::kCUDA});

  auto jit_condition = at::clone(condition);
  auto jit_x = at::clone(x);
  auto jit_y = at::clone(y);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_condition, jit_x, jit_y});

  auto trt_condition = at::clone(condition);
  auto trt_x = at::clone(x);
  auto trt_y = at::clone(y);
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_condition, trt_x, trt_y});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}