#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

std::string gen_test_graph() {
  return R"IR(
      graph(%0: Tensor):
        %3 : Tensor = aten::bitwise_not(%0)
        return (%3))IR";
}

#define test_bitwise_not(dtype)                                                      \
  TEST(Converters, ATenBitwiseNot##dtype##ConvertsCorrectly) {                       \
    const auto graph = gen_test_graph();                                             \
                                                                                     \
    auto g = std::make_shared<torch::jit::Graph>();                                  \
    torch::jit::parseIR(graph, g.get());                                             \
                                                                                     \
    at::Tensor in;                                                                   \
    if (strcmp(#dtype, "Integer") == 0)                                              \
      in = at::randint(-128, 128, {10}, {at::kCUDA}).toType(at::kInt);               \
    if (strcmp(#dtype, "Boolean") == 0)                                              \
      in = at::randint(0, 1, {10}, {at::kCUDA}).toType(at::kBool);                   \
    auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});      \
    auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});       \
                                                                                     \
    in = at::clone(in);                                                              \
    params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});           \
    auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in}); \
                                                                                     \
    auto jit_int = jit_results[0].toType(at::kInt);                                  \
    auto trt_int = trt_results[0].toType(at::kInt);                                  \
                                                                                     \
    ASSERT_TRUE(torch_tensorrt::tests::util::exactlyEqual(jit_int, trt_int));        \
  }

test_bitwise_not(Integer);
test_bitwise_not(Boolean);

#undef test_bitwise_not
