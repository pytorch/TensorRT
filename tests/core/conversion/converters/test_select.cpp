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

TEST(Converters, ATenIndexTensorConvertsCorrectly) {
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
