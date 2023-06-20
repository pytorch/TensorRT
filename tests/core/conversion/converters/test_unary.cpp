#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/torch.h"

namespace {
std::string gen_test_graph(const std::string& unary) {
  return R"IR(
      graph(%0 : Tensor):
        %3 : Tensor = aten::)IR" +
      unary + R"IR((%0)
        return (%3))IR";
}
} // namespace

TEST(Converters, ATenAbsIntConvertsCorrectly) {
  const auto graph = gen_test_graph("abs");
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::tensor({-1, 1, -2, 2, -3, 3}, {at::kCUDA}).to(torch::kInt32);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::exactlyEqual(jit_results[0], trt_results[0]));
}

TEST(Converters, ATenReciprocalIntConvertsCorrectly) {
  const auto graph = gen_test_graph("reciprocal");
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::tensor({-1, 1, -2, 2, -3, 3}, {at::kCUDA}).to(torch::kInt32);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::exactlyEqual(jit_results[0], trt_results[0]));
}

TEST(Converters, ATenLog2IntConvertsCorrectly) {
  const auto graph = gen_test_graph("log2");
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::tensor({1, 2, 7, 25, 50}, {at::kCUDA}).to(torch::kInt32);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0]));
}

TEST(Converters, ATenSignConvertsCorrectly) {
  const auto graph = gen_test_graph("sign");
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // Resize range to [-10, 10] to span negative values
  auto in = -20 * at::rand({2, 3, 5, 5}, {at::kCUDA}) + 10;
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenSignConvertsZerosCorrectly) {
  const auto graph = gen_test_graph("sign");
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // Resize range to [-1, 1] to span negative values, cast to int to include zero
  auto in = (-2 * at::rand({7, 3, 1, 5}, {at::kCUDA}) + 1).to(torch::kInt32);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenLogicalNotBoolConvertsCorrectly) {
  const auto graph = gen_test_graph("logical_not");
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());
  auto in = at::randint(0, 2, {7, 3, 1, 5}, {at::kCUDA}).to(torch::kBool);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenFiniteConvertsCorrectly) {
  const auto graph = gen_test_graph("isfinite");
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());
  auto in = at::tensor(
      {float(0),
       std::nanf(""),
       float(2),
       std::numeric_limits<float>::infinity(),
       float(4),
       -std::numeric_limits<float>::infinity()},
      {at::kCUDA});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

#define test_unary(unary, name)                                                                  \
  TEST(Converters, ATen##name##ConvertsCorrectly) {                                              \
    const auto graph = gen_test_graph(#unary);                                                   \
                                                                                                 \
    auto g = std::make_shared<torch::jit::Graph>();                                              \
    torch::jit::parseIR(graph, g.get());                                                         \
                                                                                                 \
    float offset = 0;                                                                            \
    if (strcmp(#name, "Acosh") == 0)                                                             \
      offset += 1; /*input larger than 1 for acosh*/                                             \
    auto in = at::empty({10}, {at::kCUDA}).uniform_(0 + offset, 0.5 + offset);                   \
    auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});                  \
    auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});                   \
                                                                                                 \
    in = at::clone(in);                                                                          \
    params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});                       \
    auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});             \
                                                                                                 \
    ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6)); \
  }

test_unary(cos, Cos);
test_unary(acos, Acos);
test_unary(cosh, Cosh);
test_unary(sin, Sin);
test_unary(asin, Asin);
test_unary(sinh, Sinh);
test_unary(tan, Tan);
test_unary(atan, Atan);
test_unary(abs, Abs);
test_unary(floor, Floor);
test_unary(reciprocal, Reciprocal);
test_unary(log, Log);
test_unary(log2, Log2);
test_unary(ceil, Ceil);
test_unary(sqrt, Sqrt);
test_unary(exp, Exp);
test_unary(neg, Neg);
test_unary(erf, Erf);
test_unary(asinh, Asinh);
test_unary(acosh, Acosh);
test_unary(atanh, Atanh);
test_unary(logical_not, LogicalNot);

#undef test_unary
