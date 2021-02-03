#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

namespace {
std::string gen_test_graph(const std::string& unary) {
  return R"IR(
      graph(%0 : Tensor):
        %3 : Tensor = aten::)IR" +
      unary + R"IR((%0)
        return (%3))IR";
}
} // namespace

#define test_unary(unary, name)                                                           \
  TEST(Converters, ATen##name##ConvertsCorrectly) {                                       \
    const auto graph = gen_test_graph(#unary);                                            \
                                                                                          \
    auto g = std::make_shared<torch::jit::Graph>();                                       \
    torch::jit::parseIR(graph, &*g);                                                      \
                                                                                          \
    float offset = 0;                                                                     \
    if (#name == "Acosh") offset += 1;   /*input larger than 1 for acosh*/                \
    auto in = at::empty({10}, {at::kCUDA}).uniform_(0+offset, 0.5+offset);                \
    auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});           \
    auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});                   \
                                                                                          \
    in = at::clone(in);                                                                   \
    params = trtorch::core::conversion::get_named_params(g->inputs(), {});                \
    auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});             \
                                                                                          \
    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6)); \
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
test_unary(ceil, Ceil);
test_unary(sqrt, Sqrt);
test_unary(exp, Exp);
test_unary(neg, Neg);
test_unary(erf, Erf);
test_unary(asinh, Asinh);
test_unary(acosh, Acosh);
test_unary(atanh, Atanh);

#undef test_unary
