#include <torch/torch.h>
#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ATenUnflattenDynShapeShapeCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
            %2 : int = prim::Constant[value=1]()
            %3 : int = prim::Constant[value=512]()
            %4 : int = prim::Constant[value=1]()
            %5 : int = prim::Constant[value=1]()
            %6 : int[] = prim::ListConstruct(%3, %4, %5)
            %7 : Tensor = aten::unflatten(%x.1, %2, %6)
            return (%7))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(0, 10, {1, 512}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, {in}, true);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenUnflattenDynShapeNegativeDimsShapeCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
            %2 : int = prim::Constant[value=-2]()
            %3 : int = prim::Constant[value=512]()
            %4 : int = prim::Constant[value=1]()
            %5 : int = prim::Constant[value=1]()
            %6 : int[] = prim::ListConstruct(%3, %4, %5)
            %7 : Tensor = aten::unflatten(%x.1, %2, %6)
            return (%7))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(0, 10, {1, 512, 2}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, {in}, true);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenUnflattenDynShapeITensorShapeCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
            %2 : int = prim::Constant[value=1]()
            %3 : int = aten::size(%x.1, %2)
            %4 : int = prim::Constant[value=256]()
            %5 : int = prim::Constant[value=2]()
            %6 : int[] = prim::ListConstruct(%4, %5)
            %7 : Tensor = aten::unflatten(%x.1, %2, %6)
            return (%7))IR";
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(0, 10, {1, 512, 1}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, {in}, true);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenUnflattenDynShapeITensorShapeCorrectlyFirstDim) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
            %1 : int = prim::Constant[value=0]()
            %2 : int = prim::Constant[value=1]()
            %3 : int = aten::size(%x.1, %1)
            %6 : int[] = prim::ListConstruct(%2, %2, %3, %2, %2)
            %7 : Tensor = aten::unflatten(%x.1, %1, %6)
            return (%7))IR";
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(0, 10, {64, 512, 1}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, {in}, true);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenUnflattenDynShapeITensorShapeCorrectlyLastDim) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
            %1 : int = prim::Constant[value=2]()
            %2 : int = prim::Constant[value=1]()
            %3 : int = aten::size(%x.1, %1)
            %5 : int = prim::Constant[value=2]()
            %6 : int[] = prim::ListConstruct(%3, %2, %2)
            %7 : Tensor = aten::unflatten(%x.1, %5, %6)
            return (%7))IR";
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(0, 10, {1, 512, 9}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, {in}, true);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}