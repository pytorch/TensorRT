#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ATenMaxPool1DConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=0]()
        %2 : int = prim::Constant[value=1]()
        %3 : int = prim::Constant[value=2]()
        %5 : bool = prim::Constant[value=0]()
        %6 : int[] = prim::ListConstruct(%1)
        %7 : int[] = prim::ListConstruct(%2)
        %8 : int[] = prim::ListConstruct(%3)
        %10 : Tensor = aten::max_pool1d(%0, %8, %7, %6, %7, %5)
        return (%10))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(-5, 5, {1, 1, 10}, at::kCUDA);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenMaxPool1DConvertsCorrectlyWithDynamicInput) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=0]()
        %2 : int = prim::Constant[value=1]()
        %3 : int = prim::Constant[value=2]()
        %5 : bool = prim::Constant[value=0]()
        %6 : int[] = prim::ListConstruct(%1)
        %7 : int[] = prim::ListConstruct(%2)
        %8 : int[] = prim::ListConstruct(%3)
        %10 : Tensor = aten::max_pool1d(%0, %8, %7, %6, %7, %5)
        return (%10))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(-5, 5, {1, 1, 10}, at::kCUDA);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, {in}, true);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenMaxPool2DConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=0]()
        %2 : int = prim::Constant[value=1]()
        %3 : int = prim::Constant[value=2]()
        %5 : bool = prim::Constant[value=0]()
        %6 : int[] = prim::ListConstruct(%1, %1)
        %7 : int[] = prim::ListConstruct(%2, %2)
        %8 : int[] = prim::ListConstruct(%3, %3)
        %10 : Tensor = aten::max_pool2d(%0, %8, %7, %6, %7, %5)
        return (%10))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // PyTorch MaxPool needs a 3D input
  auto in = at::randint(-5, 5, {1, 10, 10}, at::kCUDA);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenMaxPool2DConvertsCorrectlyWithDynamicInput) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=0]()
        %2 : int = prim::Constant[value=1]()
        %3 : int = prim::Constant[value=2]()
        %5 : bool = prim::Constant[value=0]()
        %6 : int[] = prim::ListConstruct(%1, %1)
        %7 : int[] = prim::ListConstruct(%2, %2)
        %8 : int[] = prim::ListConstruct(%3, %3)
        %10 : Tensor = aten::max_pool2d(%0, %8, %7, %6, %7, %5)
        return (%10))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // PyTorch MaxPool needs a 3D input
  auto in = at::randint(-5, 5, {1, 10, 10}, at::kCUDA);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, {in}, false);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenMaxPool3DConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=0]()
        %2 : int = prim::Constant[value=1]()
        %3 : int = prim::Constant[value=2]()
        %5 : bool = prim::Constant[value=0]()
        %6 : int[] = prim::ListConstruct(%1, %1, %1)
        %7 : int[] = prim::ListConstruct(%2, %2, %2)
        %8 : int[] = prim::ListConstruct(%3, %3, %3)
        %10 : Tensor = aten::max_pool3d(%0, %8, %7, %6, %7, %5)
        return (%10))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // PyTorch MaxPool needs a 3D input
  auto in = at::randint(-5, 5, {1, 3, 10, 10, 10}, at::kCUDA);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenAvgPool1DConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=0]()
        %2 : int = prim::Constant[value=1]()
        %3 : int = prim::Constant[value=2]()
        %4 : bool = prim::Constant[value=0]()
        %5 : bool = prim::Constant[value=1]()
        %6 : int[] = prim::ListConstruct(%1)
        %7 : int[] = prim::ListConstruct(%2)
        %8 : int[] = prim::ListConstruct(%3)
        %9 : None = prim::Constant()
        %10 : Tensor = aten::avg_pool1d(%0, %8, %7, %6, %4, %5)
        return (%10))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // PyTorch AvgPool needs a 3D input
  auto in = at::randint(-5, 5, {1, 1, 10}, at::kCUDA);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenAvgPool1DCeilConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=0]()
        %2 : int = prim::Constant[value=1]()
        %3 : int = prim::Constant[value=2]()
        %4 : bool = prim::Constant[value=0]()
        %5 : bool = prim::Constant[value=1]()
        %6 : int[] = prim::ListConstruct(%1)
        %7 : int[] = prim::ListConstruct(%2)
        %8 : int[] = prim::ListConstruct(%3)
        %10 : Tensor = aten::avg_pool1d(%0, %8, %7, %6, %5, %5)
        return (%10))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // PyTorch AvgPool needs a 3D input
  auto in = at::randint(-5, 5, {1, 1, 4}, at::kCUDA);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenAvgPool1DNoCountPadConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=0]()
        %2 : int = prim::Constant[value=1]()
        %3 : int = prim::Constant[value=2]()
        %4 : bool = prim::Constant[value=0]()
        %5 : bool = prim::Constant[value=1]()
        %6 : int[] = prim::ListConstruct(%1)
        %7 : int[] = prim::ListConstruct(%2)
        %8 : int[] = prim::ListConstruct(%3)
        %10 : Tensor = aten::avg_pool1d(%0, %8, %7, %6, %4, %4)
        return (%10))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // PyTorch AvgPool needs a 3D input
  auto in = at::randint(-5, 5, {1, 1, 4}, at::kCUDA);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenAvgPool2DConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=0]()
        %2 : int = prim::Constant[value=1]()
        %3 : int = prim::Constant[value=2]()
        %4 : bool = prim::Constant[value=0]()
        %5 : bool = prim::Constant[value=1]()
        %6 : int[] = prim::ListConstruct(%1, %1)
        %7 : int[] = prim::ListConstruct(%2, %2)
        %8 : int[] = prim::ListConstruct(%3, %3)
        %9 : None = prim::Constant()
        %10 : Tensor = aten::avg_pool2d(%0, %8, %7, %6, %4, %5, %9)
        return (%10))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // PyTorch AvgPool needs a 3D input
  auto in = at::randint(-5, 5, {1, 4, 4}, at::kCUDA);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenAvgPool2DCeilConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=0]()
        %2 : int = prim::Constant[value=1]()
        %3 : int = prim::Constant[value=2]()
        %4 : bool = prim::Constant[value=0]()
        %5 : bool = prim::Constant[value=1]()
        %6 : int[] = prim::ListConstruct(%1, %1)
        %7 : int[] = prim::ListConstruct(%2, %2)
        %8 : int[] = prim::ListConstruct(%3, %3)
        %9 : None = prim::Constant()
        %10 : Tensor = aten::avg_pool2d(%0, %8, %7, %6, %5, %5, %9)
        return (%10))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // PyTorch AvgPool needs a 3D input
  auto in = at::randint(-5, 5, {1, 4, 4}, at::kCUDA);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenAvgPool2DNoCountPadConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=0]()
        %2 : int = prim::Constant[value=1]()
        %3 : int = prim::Constant[value=2]()
        %4 : bool = prim::Constant[value=0]()
        %5 : bool = prim::Constant[value=1]()
        %6 : int[] = prim::ListConstruct(%1, %1)
        %7 : int[] = prim::ListConstruct(%2, %2)
        %8 : int[] = prim::ListConstruct(%3, %3)
        %9 : None = prim::Constant()
        %10 : Tensor = aten::avg_pool2d(%0, %8, %7, %6, %4, %4, %9)
        return (%10))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // PyTorch AvgPool needs a 3D input
  auto in = at::randint(-5, 5, {1, 4, 4}, at::kCUDA);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenAvgPool3DConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=0]()
        %2 : int = prim::Constant[value=1]()
        %3 : int = prim::Constant[value=2]()
        %4 : bool = prim::Constant[value=0]()
        %5 : bool = prim::Constant[value=1]()
        %6 : int[] = prim::ListConstruct(%1, %1, %1)
        %7 : int[] = prim::ListConstruct(%2, %2, %2)
        %8 : int[] = prim::ListConstruct(%3, %3, %3)
        %9 : None = prim::Constant()
        %10 : Tensor = aten::avg_pool3d(%0, %8, %7, %6, %4, %5, %9)
        return (%10))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // PyTorch AvgPool needs a 3D input
  auto in = at::randint(-5, 5, {1, 3, 4, 4, 4}, at::kCUDA);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenAvgPool3DCeilConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=0]()
        %2 : int = prim::Constant[value=1]()
        %3 : int = prim::Constant[value=2]()
        %4 : bool = prim::Constant[value=0]()
        %5 : bool = prim::Constant[value=1]()
        %6 : int[] = prim::ListConstruct(%1, %1, %1)
        %7 : int[] = prim::ListConstruct(%2, %2, %2)
        %8 : int[] = prim::ListConstruct(%3, %3, %3)
        %9 : None = prim::Constant()
        %10 : Tensor = aten::avg_pool3d(%0, %8, %7, %6, %5, %5, %9)
        return (%10))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // PyTorch AvgPool needs a 3D input
  auto in = at::randint(-5, 5, {1, 3, 4, 4, 4}, at::kCUDA);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenAvgPool3DNoCountPadConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=0]()
        %2 : int = prim::Constant[value=1]()
        %3 : int = prim::Constant[value=2]()
        %4 : bool = prim::Constant[value=0]()
        %5 : bool = prim::Constant[value=1]()
        %6 : int[] = prim::ListConstruct(%1, %1, %1)
        %7 : int[] = prim::ListConstruct(%2, %2, %2)
        %8 : int[] = prim::ListConstruct(%3, %3, %3)
        %9 : None = prim::Constant()
        %10 : Tensor = aten::avg_pool3d(%0, %8, %7, %6, %4, %4, %9)
        return (%10))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // PyTorch AvgPool needs a 3D input
  auto in = at::randint(-5, 5, {1, 3, 4, 4, 4}, at::kCUDA);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenAdaptiveAvgPool2DConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=7]()
        %3 : int = prim::Constant[value=7]()
        %6 : int[] = prim::ListConstruct(%2, %3)
        %10 : Tensor = aten::adaptive_avg_pool2d(%0, %6)
        return (%10))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // PyTorch MaxPool needs a 3D input
  auto in = at::randint(-5, 5, {512, 32, 32}, at::kCUDA);

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenAdaptiveAvgPool2DGlobalPoolingConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=1]()
        %3 : int = prim::Constant[value=1]()
        %6 : int[] = prim::ListConstruct(%2, %3)
        %10 : Tensor = aten::adaptive_avg_pool2d(%0, %6)
        return (%10))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // PyTorch PyTorch adaptive_avg_pool2d needs a 4D input or a 3D input
  auto in = at::randint(-5, 5, {64, 16, 32, 32}, at::kCUDA);

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenAdaptiveAvgPool2DConvertsCorrectlyWithDynamicInput) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=3]()
        %3 : int = prim::Constant[value=4]()
        %6 : int[] = prim::ListConstruct(%2, %3)
        %10 : Tensor = aten::adaptive_avg_pool2d(%0, %6)
        return (%10))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // PyTorch MaxPool needs a 3D input
  auto in = at::randint(-5, 5, {512, 32, 32}, at::kCUDA);

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, {trt_in}, false);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenAdaptiveAvgPool1DConvertsCorrectly) {
  const auto graph =
      R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=1]()
        %6 : int[] = prim::ListConstruct(%2)
        %10 : Tensor = aten::adaptive_avg_pool1d(%0, %6)
        return (%10))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // PyTorch MaxPool needs a 3D input
  auto in = at::randint(-5, 5, {1, 3, 16}, at::kCUDA);

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 1.0));
}

TEST(Converters, ATenAdaptiveAvgPool1DGlobalPoolingConvertsCorrectly) {
  const auto graph =
      R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=1]()
        %6 : int[] = prim::ListConstruct(%2)
        %10 : Tensor = aten::adaptive_avg_pool1d(%0, %6)
        return (%10))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // PyTorch adaptive_avg_pool1d needs a 3D input or a 2D input
  auto in = at::randint(-5, 5, {3, 16}, at::kCUDA);

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenAdaptiveAvgPool1DUsingPluginConvertsCorrectly) {
  const auto graph =
      R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=3]()
        %6 : int[] = prim::ListConstruct(%2)
        %10 : Tensor = aten::adaptive_avg_pool1d(%0, %6)
        return (%10))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // PyTorch adaptive_avg_pool1d needs a 3D input or a 2D input
  auto in = at::randint(-5, 5, {1, 3, 16}, at::kCUDA);

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenAdaptiveMaxPool1DGlobalPoolingConvertsCorrectly) {
  const auto graph =
      R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=1]()
        %6 : int[] = prim::ListConstruct(%2)
        %10 : Tensor, %11 : Tensor = aten::adaptive_max_pool1d(%0, %6)
        return (%10, %11))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // PyTorch adaptive_max_pool1d needs a 3D input or a 2D input
  auto in = at::randint(-5, 5, {1, 3, 16}, at::kCUDA);

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenAdaptiveMaxPool1DUsingPluginConvertsCorrectly) {
  const auto graph =
      R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=3]()
        %6 : int[] = prim::ListConstruct(%2)
        %10 : Tensor, %11 : Tensor = aten::adaptive_max_pool1d(%0, %6)
        return (%10, %11))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // PyTorch adaptive_max_pool1d needs a 3D input or a 2D input
  auto in = at::randint(-5, 5, {1, 3, 16}, at::kCUDA);

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenAdaptiveMaxPool2DConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=7]()
        %3 : int = prim::Constant[value=7]()
        %6 : int[] = prim::ListConstruct(%2, %3)
        %10 : Tensor, %11 : Tensor = aten::adaptive_max_pool2d(%0, %6)
        return (%10, %11))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  // PyTorch MaxPool needs a 3D input
  auto in = at::randint(-5, 5, {512, 32, 32}, at::kCUDA);

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenAdaptiveMaxPool2DConvertsCorrectlyWithDynamicInput) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=7]()
        %3 : int = prim::Constant[value=7]()
        %6 : int[] = prim::ListConstruct(%2, %3)
        %10 : Tensor, %11 : Tensor = aten::adaptive_max_pool2d(%0, %6)
        return (%10, %11))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  // PyTorch MaxPool needs a 3D input
  auto in = at::rand({512, 32, 32}, at::kCUDA);

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, {trt_in}, false);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenAdaptiveAvgPool3DGlobalPoolingConvertsCorrectly) {
  const auto graph =
      R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=1]()
        %3 : int = prim::Constant[value=1]()
        %4 : int = prim::Constant[value=1]()
        %6 : int[] = prim::ListConstruct(%2, %3, %4)
        %10 : Tensor = aten::adaptive_avg_pool3d(%0, %6)
        return (%10))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // PyTorch adaptive_avg_pool3d needs a 5D input or a 4D input
  auto in = at::randint(-5, 5, {4, 5, 3, 15, 16}, at::kCUDA);

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenAdaptiveAvgPool3DUsingPluginConvertsCorrectly) {
  const auto graph =
      R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=7]()
        %3 : int = prim::Constant[value=6]()
        %4 : int = prim::Constant[value=5]()
        %6 : int[] = prim::ListConstruct(%2, %3, %4)
        %10 : Tensor = aten::adaptive_avg_pool3d(%0, %6)
        return (%10))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // PyTorch adaptive_avg_pool3d needs a 5D input or a 4D input
  auto in = at::randint(-5, 5, {4, 5, 3, 15, 16}, at::kCUDA);

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenAdaptiveMaxPool3DGlobalPoolingConvertsCorrectly) {
  const auto graph =
      R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=1]()
        %3 : int = prim::Constant[value=1]()
        %4 : int = prim::Constant[value=1]()
        %6 : int[] = prim::ListConstruct(%2, %3, %4)
        %10 : Tensor, %11 : Tensor = aten::adaptive_max_pool3d(%0, %6)
        return (%10, %11))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // PyTorch adaptive_max_pool3d needs a 5D input or a 4D input
  auto in = at::randint(-5, 5, {5, 3, 15, 16}, at::kCUDA);

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenAdaptiveMaxPool3DUsingPluginConvertsCorrectly) {
  const auto graph =
      R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=7]()
        %3 : int = prim::Constant[value=8]()
        %4 : int = prim::Constant[value=9]()
        %6 : int[] = prim::ListConstruct(%2, %3, %4)
        %10 : Tensor, %11 : Tensor = aten::adaptive_max_pool3d(%0, %6)
        return (%10, %11))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // PyTorch adaptive_max_pool3d needs a 5D input or a 4D input
  auto in = at::randint(-5, 5, {4, 5, 3, 15, 16}, at::kCUDA);

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}
