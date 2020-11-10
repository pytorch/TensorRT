#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ATenSelectIntConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=0]()
        %3 : Tensor = aten::select(%0, %2, %2)
        return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {4, 4, 4}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenSelectIntTwiceConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=0]()
        %3 : int = prim::Constant[value=3]()
        %4 : Tensor = aten::select(%0, %2, %2)
        %5 : Tensor = aten::select(%4, %2, %3)
        return (%5))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {4, 4, 4}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenNarrowStartScalarConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
            %2 : int = prim::Constant[value=2]()
            %3 : int = prim::Constant[value=0]()
            %4 : Tensor = aten::narrow(%x.1, %3, %3, %2)
            return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {3, 2, 2, 4}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenEmbeddingConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%1 : Tensor, %emb_weight : Float(10:3, 3:1)):
            %2 : bool = prim::Constant[value=0]()
            %3 : int = prim::Constant[value=-1]()
            %5 : Tensor = aten::embedding(%emb_weight, %1, %3, %2, %2)
            return (%5))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  // Run Pytorch
  torch::jit::parseIR(graph, &*g);
  auto options_pyt = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kLong);
  auto jit_in = at::tensor({0, 1, 2}, options_pyt);
  auto embWeight = at::randn({10, 3}, {at::kCUDA});

  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {embWeight});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

  // Run TensorRT
  auto options_trt = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kI32);
  auto trt_in = at::tensor({0, 1, 2}, options_trt);
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in});
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}