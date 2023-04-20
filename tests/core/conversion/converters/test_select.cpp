#include <torch/torch.h>
#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
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
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
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
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  // In order to check whether shape match that we don't do reshape.
  // E.g. x = at::randint(1, 10, {4, 4, 4}, {at::kCUDA}), then aten::select(x, 1, 0). We should get a tensor y with
  // shape {4, 4} instead of a tensor with shape {4, 1, 4}.
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
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
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenSelectIntNegIndexConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=0]()
        %3 : int = prim::Constant[value=-1]()
        %4 : Tensor = aten::select(%0, %3, %2)
        return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = torch::tensor({2, 20, 768}).to(at::kFloat).to(at::kCUDA);

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
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
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenSelectEmptyTensorConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=1]()
        %3 : int = prim::Constant[value=0]()
        %4 : Tensor = aten::select(%0, %3, %2)
        return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = torch::ones({2, 20, 0, 768}).to(at::kCUDA);

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  ASSERT_TRUE(torch_tensorrt::tests::util::sameShape(jit_results[0], trt_results[0]));
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
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenEmbeddingConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%1 : Tensor, %emb_weight : Float(10, 3, strides=[3, 1])):
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

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {embWeight});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  // Run TensorRT
  auto options_trt = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kFloat);
  auto trt_in = at::tensor({0, 1, 2}, options_trt);
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}
