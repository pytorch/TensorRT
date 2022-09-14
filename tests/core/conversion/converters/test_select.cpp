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

TEST(Converters, ATenRollConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%1 : Tensor):
            %2 : int[] = prim::Constant[value=[1, 0, 3, 7]]()
            %3 : int[] = prim::Constant[value=[0, 1, 2, 3]]()
            %4 : Tensor = aten::roll(%1, %2, %3)
            return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  // Run Pytorch
  auto in = at::randint(1, 10, {2, 3, 4, 5}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenRollShiftsNegativeConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%1 : Tensor):
            %2 : int[] = prim::Constant[value=[0, -3, -3]]()
            %3 : int[] = prim::Constant[value=[1, 2, 3]]()
            %4 : Tensor = aten::roll(%1, %2, %3)
            return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  // Run Pytorch
  auto in = at::randint(1, 10, {1, 3, 4, 5}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenRollDimsNegativeConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%1 : Tensor):
            %2 : int[] = prim::Constant[value=[0, -3, -3]]()
            %3 : int[] = prim::Constant[value=[1, 2, -1]]()
            %4 : Tensor = aten::roll(%1, %2, %3)
            return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  // Run Pytorch
  auto in = at::randint(1, 10, {1, 3, 4, 5}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenSliceConvertsCorrectly) {
  const auto graph = R"IR(
        graph(%x.1 : Tensor):
              %2 : None = prim::Constant()
              %3 : int = prim::Constant[value=2]()
              %4 : int = prim::Constant[value=4]()
              %5 : int = prim::Constant[value=1]()
              %6 : int = prim::Constant[value=0]()
              %7 : Tensor = aten::select(%x.1, %6, %6)
              %8 : Tensor = aten::select(%7, %6, %5)
              %9 : Tensor = aten::slice(%8, %6, %5, %4, %3)
              %10 : Tensor = aten::slice(%9, %5, %2, %2, %5)
              return (%10))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {1, 3, 5, 5}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
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
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
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
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenSliceListConvertsCorrectly) {
  const auto graph = R"IR(
        graph(%x : Tensor):
          %1 : NoneType = prim::Constant()
          %2 : int = prim::Constant[value=2]()
          %3 : int = prim::Constant[value=1]()
          %4 : int = prim::Constant[value=3]()
          %list : Tensor[] = aten::unbind(%x, %4)
          %slice : Tensor[] = aten::slice(%list, %1, %2, %3)
          %out.1 : Tensor, %out.2 : Tensor = prim::ListUnpack(%slice)
          return (%out.1, %out.2))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in_x = at::randint(1, 10, {6, 5, 3, 3}, {at::kCUDA});

  auto jit_in_x = at::clone(in_x);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in_x});

  auto trt_in_x = at::clone(in_x);
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in_x});

  for (size_t i = 0; i < jit_results.size(); i++) {
    auto trt = trt_results[i].reshape(jit_results[i].sizes());
    ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[i], trt, 2e-6));
  }
}

TEST(Converters, ATenSliceDynamicBatchConvertsCorrectly) {
  const auto graph = R"IR(
        graph(%x.1 : Tensor):
              %2 : None = prim::Constant()
              %dim : int = prim::Constant[value=0]()
              %start : int = prim::Constant[value=1]()
              %end : int = prim::Constant[value=15]()
              %step : int = prim::Constant[value=2]()
              %9 : Tensor = aten::slice(%x.1, %dim, %start, %end, %step)
              return (%9))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {16, 32}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  // dynamic shape in batch
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, {trt_in}, true);
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenSliceDynamicBatchLargeEndConvertsCorrectly) {
  const auto graph = R"IR(
        graph(%x.1 : Tensor):
              %2 : None = prim::Constant()
              %dim : int = prim::Constant[value=0]()
              %start : int = prim::Constant[value=1]()
              %end : int = prim::Constant[value=9223372036854775807]()
              %step : int = prim::Constant[value=2]()
              %9 : Tensor = aten::slice(%x.1, %dim, %start, %end, %step)
              return (%9))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {16, 32}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  // dynamic shape in batch
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, {trt_in}, true);
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenSliceDynamicNegStartBatchConvertsCorrectly) {
  const auto graph = R"IR(
        graph(%x.1 : Tensor):
              %2 : None = prim::Constant()
              %dim : int = prim::Constant[value=0]()
              %start : int = prim::Constant[value=-15]()
              %end : int = prim::Constant[value=15]()
              %step : int = prim::Constant[value=2]()
              %9 : Tensor = aten::slice(%x.1, %dim, %start, %end, %step)
              return (%9))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {16, 32}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  // dynamic shape in batch
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, {trt_in}, true);
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenSliceDynamicNegEndBatchConvertsCorrectly) {
  const auto graph = R"IR(
        graph(%x.1 : Tensor):
              %2 : None = prim::Constant()
              %dim : int = prim::Constant[value=0]()
              %start : int = prim::Constant[value=1]()
              %end : int = prim::Constant[value=-2]()
              %step : int = prim::Constant[value=3]()
              %9 : Tensor = aten::slice(%x.1, %dim, %start, %end, %step)
              return (%9))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {16, 32}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  // dynamic shape in batch
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, {trt_in}, true);
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenSliceDynamicNoneBatchConvertsCorrectly) {
  const auto graph = R"IR(
        graph(%x.1 : Tensor):
              %dim : int = prim::Constant[value=0]()
              %start : None = prim::Constant()
              %end : None = prim::Constant()
              %step : int = prim::Constant[value=3]()
              %9 : Tensor = aten::slice(%x.1, %dim, %start, %end, %step)
              return (%9))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {16, 32}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  // dynamic shape in batch
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, {trt_in}, true);
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenSliceDynamicConvertsCorrectly) {
  const auto graph = R"IR(
        graph(%x.1 : Tensor):
              %2 : None = prim::Constant()
              %dim : int = prim::Constant[value=1]()
              %start : int = prim::Constant[value=3]()
              %end : int = prim::Constant[value=32]()
              %step : int = prim::Constant[value=3]()
              %9 : Tensor = aten::slice(%x.1, %dim, %start, %end, %step)
              return (%9))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {16, 32}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  // dynamic shape in dim 1, slice in dim 1
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, {trt_in}, false);
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenSliceDynamic2ConvertsCorrectly) {
  const auto graph = R"IR(
        graph(%x.1 : Tensor):
              %2 : None = prim::Constant()
              %dim : int = prim::Constant[value=1]()
              %start : int = prim::Constant[value=3]()
              %end : int = prim::Constant[value=17]()
              %step : int = prim::Constant[value=3]()
              %9 : Tensor = aten::slice(%x.1, %dim, %start, %end, %step)
              return (%9))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {16, 32}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  // dynamic shape in batch, slice in dim 1
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, {trt_in}, true);
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
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
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  for (size_t i = 0; i < jit_results.size(); i++) {
    auto trt = trt_results[i].reshape(jit_results[i].sizes());
    ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[i], trt, 2e-6));
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
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  for (size_t i = 0; i < jit_results.size(); i++) {
    auto trt = trt_results[i].reshape(jit_results[i].sizes());
    ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[i], trt, 2e-6));
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
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  for (size_t i = 0; i < jit_results.size(); i++) {
    auto trt = trt_results[i].reshape(jit_results[i].sizes());
    ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[i], trt, 2e-6));
  }
}

TEST(Converters, ATenSplitFixedHasRemainderConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%argument_1.1 : Tensor):
          %2 : int = prim::Constant[value=2]()
          %2.1 : int = prim::Constant[value=1]()
          %3 : Tensor[] = aten::split(%argument_1.1, %2, %2.1)
          %4 : Tensor, %5 : Tensor, %6 : Tensor = prim::ListUnpack(%3)
          return (%4, %5, %6))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {1, 5, 4, 4}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  for (size_t i = 0; i < jit_results.size(); i++) {
    auto trt = trt_results[i].reshape(jit_results[i].sizes());
    ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[i], trt, 2e-6));
  }
}

TEST(Converters, ATenSplitAndAddConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%argument_1.1 : Tensor):
          %2 : int = prim::Constant[value=2]()
          %2.1 : int = prim::Constant[value=1]()
          %3 : Tensor[] = aten::split(%argument_1.1, %2, %2.1)
          %4 : Tensor, %5 : Tensor = prim::ListUnpack(%3)
          %6 : Tensor = aten::add(%4, %5, %2.1)
          return (%6))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {1, 4, 4, 4}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  for (size_t i = 0; i < jit_results.size(); i++) {
    auto trt = trt_results[i].reshape(jit_results[i].sizes());
    ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[i], trt, 2e-6));
  }
}

TEST(Converters, ATenMaskedFillZerosConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
      %44 : Device = prim::Constant[value="cuda"]()
      %8 : bool = prim::Constant[value=0]()
      %7 : None = prim::Constant()
      %f32_dtype: int = prim::Constant[value=11]()
      %1 : int = prim::Constant[value=0]() # bert.py:5:26
      %2 : int = prim::Constant[value=1]() # bert.py:5:32
      %33 : int = prim::Constant[value=2]() # bert.py:6:31
      %3 : int[] = prim::ListConstruct(%1, %1, %2)
      %4 : int[] = prim::ListConstruct(%2, %2, %1)
      %5 : int[][] = prim::ListConstruct(%3, %4)
      %9 : Tensor = aten::tensor(%5, %f32_dtype, %7, %8) # bert.py:5:11
      %mask.1 : Tensor = aten::to(%9, %44, %7, %8, %8) # bert.py:5:11
      %mask.2 : Tensor = trt::const(%mask.1)
      %34 : Tensor = aten::masked_fill(%x.1, %mask.1, %33) # bert.py:6:11
      return (%34, %mask.2))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  auto in = at::zeros({1, 2, 3}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  torch_tensorrt::core::lowering::passes::RemoveNOPs(g);
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenIndexTensorOneIndiceConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor,
            %index : Tensor):
        %18 : Tensor?[] = prim::ListConstruct(%index)
        %19 : Tensor = aten::index(%x.1, %18)
        return (%19))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in1 = at::randint(1, 10, {5, 10}, {at::kCUDA});
  auto in2 = at::full({2}, 4, {at::kCUDA});
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto in2_trt = at::full({2}, 4, {options});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in1, in2});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in1, in2_trt});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenIndexTensorFullIndicesConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor,
            %index0 : Tensor,
            %index1 : Tensor,
            %index2 : Tensor):
        %18 : Tensor?[] = prim::ListConstruct(%index0, %index1, %index2)
        %19 : Tensor = aten::index(%x.1, %18)
        return (%19))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in1 = at::randint(1, 10, {5, 10, 4}, {at::kCUDA});
  auto index0 = at::tensor({0, 1, 2, 3}, {at::kCUDA}).to(torch::kLong);
  auto index1 = at::tensor({1, 3, 4, 6}, {at::kCUDA}).to(torch::kLong);
  auto index2 = at::tensor({3, 2, 1, 0}, {at::kCUDA}).to(torch::kLong);
  auto index0_trt = index0.to(torch::kInt32);
  auto index1_trt = index1.to(torch::kInt32);
  auto index2_trt = index2.to(torch::kInt32);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in1, index0, index1, index2});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in1, index0_trt, index1_trt, index2_trt});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenIndexTensorIdx0Idx1NoneConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor,
            %index0 : Tensor,
            %index1 : Tensor):
        %5 : NoneType = prim::Constant()
        %18 : Tensor?[] = prim::ListConstruct(%index0, %index1, %5)
        %19 : Tensor = aten::index(%x.1, %18)
        return (%19))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in1 = at::randint(1, 10, {5, 10, 4}, {at::kCUDA});
  auto index0 = at::tensor({0, 1, 2, 3}, {at::kCUDA}).to(torch::kLong);
  auto index1 = at::tensor({1, 3, 4, 6}, {at::kCUDA}).to(torch::kLong);
  auto index0_trt = index0.to(torch::kInt32);
  auto index1_trt = index1.to(torch::kInt32);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in1, index0, index1});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in1, index0_trt, index1_trt});
  LOG_DEBUG(trt_results);

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenIndexTensorIdx0NoneIdx1ConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor,
            %index0 : Tensor,
            %index1 : Tensor):
        %5 : NoneType = prim::Constant()
        %18 : Tensor?[] = prim::ListConstruct(%index0, %5, %index1)
        %19 : Tensor = aten::index(%x.1, %18)
        return (%19))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in1 = at::randint(1, 10, {5, 10, 4}, {at::kCUDA});
  auto index0 = at::tensor({0, 1, 2, 3}, {at::kCUDA}).to(torch::kLong);
  auto index1 = at::tensor({3, 2, 1, 0}, {at::kCUDA}).to(torch::kLong);
  auto index0_trt = index0.to(torch::kInt32);
  auto index1_trt = index1.to(torch::kInt32);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in1, index0, index1});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in1, index0_trt, index1_trt});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenIndexTensorNoneIdx0Idx1ConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor,
            %index0 : Tensor,
            %index1 : Tensor):
        %5 : NoneType = prim::Constant()
        %18 : Tensor?[] = prim::ListConstruct(%5, %index0, %index1)
        %19 : Tensor = aten::index(%x.1, %18)
        return (%19))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in1 = at::randint(1, 10, {5, 10, 4}, {at::kCUDA});
  auto index0 = at::tensor({0, 1, 2, 3}, {at::kCUDA}).to(torch::kLong);
  auto index1 = at::tensor({3, 2, 1, 0}, {at::kCUDA}).to(torch::kLong);
  auto index0_trt = index0.to(torch::kInt32);
  auto index1_trt = index1.to(torch::kInt32);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in1, index0, index1});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in1, index0_trt, index1_trt});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenIndexTensorIdxsNoneConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor,
            %index0 : Tensor,
            %index1 : Tensor,
            %index2 : Tensor):
        %5 : NoneType = prim::Constant()
        %18 : Tensor?[] = prim::ListConstruct(%index0, %index1, %index2, %5)
        %19 : Tensor = aten::index(%x.1, %18)
        return (%19))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in1 = at::randint(1, 10, {4, 8, 8, 4}, {at::kCUDA});
  auto index0 = at::full({4, 13, 1}, 1, {at::kCUDA}).to(torch::kLong);
  auto index1 = at::full({4, 13, 1}, 2, {at::kCUDA}).to(torch::kLong);
  auto index2 = at::full({4, 13, 1}, 3, {at::kCUDA}).to(torch::kLong);
  auto index0_trt = index0.to(torch::kInt32);
  auto index1_trt = index1.to(torch::kInt32);
  auto index2_trt = index2.to(torch::kInt32);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in1, index0, index1, index2});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in1, index0_trt, index1_trt, index2_trt});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

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
    auto trt = trt_results[i].reshape(jit_results[i].sizes());
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
    auto trt = trt_results[i].reshape(jit_results[i].sizes());
    ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[i], trt, 2e-6));
  }
}

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