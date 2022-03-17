#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

// TODO: IR Parser doesnt work well with neg numbers
TEST(Converters, ATenFlattenConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor):
      %1 : int = prim::Constant[value=0]()
      %2 : int = prim::Constant[value=1]()
      %3 : Tensor = aten::flatten(%0, %1, %2)
      return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(0, 5, {2, 3}, {at::kCUDA});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  auto trt = trt_results[0].reshape_as(jit_results[0]);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

// TODO: IR Parser doesnt work well with neg numbers
TEST(Converters, ATenFlattenOtherDimsConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor):
      %1 : int = prim::Constant[value=1]()
      %2 : int = prim::Constant[value=2]()
      %3 : Tensor = aten::flatten(%0, %1, %2)
      return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(0, 5, {2, 3, 3}, {at::kCUDA});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  auto trt = trt_results[0].reshape_as(jit_results[0]);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenReshapeConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor):
      %1 : int = prim::Constant[value=3]()
      %2 : int = prim::Constant[value=2]()
      %3 : int[] = prim::ListConstruct(%1, %2)
      %4 : Tensor = aten::reshape(%0, %3)
      return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(0, 5, {2, 3}, {at::kCUDA});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  auto trt = trt_results[0].reshape_as(jit_results[0]);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenViewConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor):
      %1 : int = prim::Constant[value=3]()
      %2 : int = prim::Constant[value=2]()
      %3 : int[] = prim::ListConstruct(%1, %2)
      %4 : Tensor = aten::view(%0, %3)
      return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(0, 5, {2, 3}, {at::kCUDA});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  auto trt = trt_results[0].reshape_as(jit_results[0]);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenPermuteConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
      %2 : int[] = prim::Constant[value=[3, 0, 1, 2]]()
      %3 : Tensor = aten::permute(%x.1, %2)
      return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(0, 5, {2, 3, 2, 3}, {at::kCUDA});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});

  std::cout << "Running JIT" << std::endl;
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  std::cout << "Running TRT" << std::endl;
  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  auto trt = trt_results[0].reshape_as(jit_results[0]);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenPermute3DConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
      %2 : int[] = prim::Constant[value=[0, 2, 1]]()
      %3 : Tensor = aten::permute(%x.1, %2)
      return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(0, 5, {2, 2, 3}, {at::kCUDA});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  auto trt = trt_results[0].reshape_as(jit_results[0]);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenPermute5DConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
      %2 : int[] = prim::Constant[value=[3, 4, 0, 2, 1]]()
      %3 : Tensor = aten::permute(%x.1, %2)
      return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(0, 5, {2, 2, 1, 2, 3}, {at::kCUDA});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  auto trt = trt_results[0].reshape_as(jit_results[0]);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenFlattenConvertsCorrectlyWithDynamicInput) {
  const auto graph = R"IR(
    graph(%0 : Tensor):
      %1 : int = prim::Constant[value=0]()
      %2 : int = prim::Constant[value=1]()
      %3 : Tensor = aten::flatten(%0, %1, %2)
      return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(0, 5, {2, 3}, {at::kCUDA});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, {in}, false);
  auto trt = trt_results[0].reshape_as(jit_results[0]);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenFlattenConvertsCorrectlyWithDynamicBatch) {
  const auto graph = R"IR(
    graph(%0 : Tensor):
      %1 : int = prim::Constant[value=0]()
      %2 : int = prim::Constant[value=1]()
      %3 : Tensor = aten::flatten(%0, %1, %2)
      return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(0, 5, {2, 3}, {at::kCUDA});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, {in}, true);
  auto trt = trt_results[0].reshape_as(jit_results[0]);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenTransposeConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
      %2 : int = prim::Constant[value=1]()
      %3 : int = prim::Constant[value=3]()
      %4 : Tensor = aten::transpose(%x.1, %2, %3)
      return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(0, 5, {2, 3, 4, 5, 6}, {at::kCUDA});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});

  std::cout << "Running JIT" << std::endl;
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  std::cout << "Running TRT" << std::endl;
  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  auto trt = trt_results[0].reshape_as(jit_results[0]);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenTConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
      %out : Tensor = aten::t(%x.1)
      return (%out))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(0, 5, {3, 4}, {at::kCUDA});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});

  std::cout << "Running JIT" << std::endl;
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  std::cout << "Running TRT" << std::endl;
  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  auto trt = trt_results[0].reshape_as(jit_results[0]);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenTransposeNegativeConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
      %2 : int = prim::Constant[value=-1]()
      %3 : int = prim::Constant[value=-3]()
      %4 : Tensor = aten::transpose(%x.1, %2, %3)
      return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(0, 5, {2, 3, 4, 5, 6}, {at::kCUDA});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});

  std::cout << "Running JIT" << std::endl;
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  std::cout << "Running TRT" << std::endl;
  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  auto trt = trt_results[0].reshape_as(jit_results[0]);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenPixelShuffleConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
      %2 : int = prim::Constant[value=3]()
      %3 : Tensor = aten::pixel_shuffle(%x.1, %2)
      return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(0, 5, {1, 9, 4, 5}, {at::kCUDA});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});

  std::cout << "Running JIT" << std::endl;
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  std::cout << "Running TRT" << std::endl;
  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  // auto trt = trt_results[0].reshape_as(jit_results[0]);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenPixelShuffle3DConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
      %2 : int = prim::Constant[value=3]()
      %3 : Tensor = aten::pixel_shuffle(%x.1, %2)
      return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(0, 5, {9, 4, 5}, {at::kCUDA});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});

  std::cout << "Running JIT" << std::endl;
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  std::cout << "Running TRT" << std::endl;
  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenPixelShuffle5DConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
      %2 : int = prim::Constant[value=3]()
      %3 : Tensor = aten::pixel_shuffle(%x.1, %2)
      return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(0, 5, {2, 3, 9, 4, 5}, {at::kCUDA});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});

  std::cout << "Running JIT" << std::endl;
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  std::cout << "Running TRT" << std::endl;
  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  // auto trt = trt_results[0].reshape_as(jit_results[0]);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}
