#include <torch/torch.h>
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

  torch::jit::parseIR(graph, g.get());

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

TEST(Converters, ATenSelectIntDimIsOneConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=1]()
        %3 : int = prim::Constant[value=0]()
        %4 : Tensor = aten::select(%0, %2, %3)
        return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {4, 4, 4}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in});

  // In order to check whether shape match that we don't do reshape.
  // E.g. x = at::randint(1, 10, {4, 4, 4}, {at::kCUDA}), then aten::select(x, 1, 0). We should get a tensor y with
  // shape {4, 4} instead of a tensor with shape {4, 1, 4}.
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenSelectIntDimNegativeConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=-2]()
        %3 : int = prim::Constant[value=0]()
        %4 : Tensor = aten::select(%0, %2, %3)
        return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {4, 4, 4}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
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

  torch::jit::parseIR(graph, g.get());

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

  torch::jit::parseIR(graph, g.get());

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
  torch::jit::parseIR(graph, g.get());
  auto options_pyt = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kLong);
  auto jit_in = at::tensor({0, 1, 2}, options_pyt);
  auto embWeight = at::randn({10, 3}, {at::kCUDA});

  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {embWeight});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

  // Run TensorRT
  auto options_trt = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kFloat);
  auto trt_in = at::tensor({0, 1, 2}, options_trt);
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in});
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenSliceConvertsCorrectly) {
  const auto graph = R"IR(
        graph(%x.1 : Tensor):
              %2 : int = prim::Constant[value=9223372036854775807]()
              %3 : int = prim::Constant[value=2]()
              %4 : int = prim::Constant[value=4]()
              %5 : int = prim::Constant[value=1]()
              %6 : int = prim::Constant[value=0]()
              %7 : Tensor = aten::select(%x.1, %6, %6)
              %8 : Tensor = aten::select(%7, %6, %5)
              %9 : Tensor = aten::slice(%8, %6, %5, %4, %3)
              %10 : Tensor = aten::slice(%9, %5, %6, %2, %5)
              return (%10))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {1, 3, 5, 5}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in});
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenSliceNegStartIndexConvertsCorrectly) {
  const auto graph = R"IR(
        graph(%x.1 : Tensor):
              %2 : int = prim::Constant[value=1]()
              %3 : int = prim::Constant[value=9223372036854775807]()
              %4 : int = prim::Constant[value=-2]()
              %5 : int = prim::Constant[value=0]()
              %6 : Tensor = aten::slice(%x.1, %5, %4, %3, %2)
              %7 : Tensor = aten::slice(%6, %2, %5, %3, %2)
              return (%7))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {6, 3}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in});
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenSliceNegEndIndexConvertsCorrectly) {
  const auto graph = R"IR(
        graph(%x.1 : Tensor):
              %2 : int = prim::Constant[value=3]()
              %3 : int = prim::Constant[value=9223372036854775807]()
              %4 : int = prim::Constant[value=2]()
              %5 : int = prim::Constant[value=-3]()
              %6 : int = prim::Constant[value=1]()
              %7 : int = prim::Constant[value=-2]()
              %8 : int = prim::Constant[value=0]()
              %9 : Tensor = aten::slice(%x.1, %8, %8, %7, %6)
              %10 : Tensor = aten::slice(%9, %6, %8, %5, %6)
              %11 : Tensor = aten::slice(%10, %4, %8, %3, %6)
              %12 : Tensor = aten::slice(%11, %2, %8, %3, %6)
              return (%12))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {6, 5, 3, 3}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in});
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenSplitSizesInScriptingConvertsCorrectly) {
  const auto graph = R"IR(
        graph(%x.1 : Tensor):
              %2 : int[] = prim::Constant[value=[1, 2]]()
              %3 : int = prim::Constant[value=1]()
              %4 : Tensor[] = aten::split(%x.1, %2, %3)
              %x1.1 : Tensor, %x2.1 : Tensor = prim::ListUnpack(%4)
              return (%x1.1, %x2.1))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {1, 3, 4, 4}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in});

  for (size_t i = 0; i < jit_results.size(); i++) {
    auto trt = trt_results[i].reshape(jit_results[i].sizes());
    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[i], trt, 2e-6));
  }
}

TEST(Converters, ATenSplitSizesinTracingConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%argument_1.1 : Tensor):
            %2 : int[] = prim::Constant[value=[1, 2]]()
            %3 : int = prim::Constant[value=1]()
            %4 : Tensor[] = aten::split_with_sizes(%argument_1.1, %2, %3)
            %5 : Tensor, %6 : Tensor = prim::ListUnpack(%4)
            return (%5, %6))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {1, 3, 4, 4}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in});

  for (size_t i = 0; i < jit_results.size(); i++) {
    auto trt = trt_results[i].reshape(jit_results[i].sizes());
    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[i], trt, 2e-6));
  }
}

TEST(Converters, ATenSplitFixedConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%argument_1.1 : Tensor):
          %2 : int = prim::Constant[value=1]()
          %3 : Tensor[] = aten::split(%argument_1.1, %2, %2)
          %4 : Tensor, %5 : Tensor, %6 : Tensor = prim::ListUnpack(%3)
          return (%4, %5, %6))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {1, 3, 4, 4}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in});

  for (size_t i = 0; i < jit_results.size(); i++) {
    auto trt = trt_results[i].reshape(jit_results[i].sizes());
    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[i], trt, 2e-6));
  }
}
