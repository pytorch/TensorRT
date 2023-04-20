#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ATenIndexSelectConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor, %index : Int (2)):
        %2 : int = prim::Constant[value=0]()
        %3 : Tensor = aten::index_select(%0, %2, %index)
        return (%3))IR";
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());
  auto in = at::randint(1, 10, {4, 4, 4}, {at::kCUDA});
  auto index = at::randint(0, 4, {2}, {at::kCUDA}).to(torch::kI32);

  auto jit_in = at::clone(in);
  auto jit_index = at::clone(index);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {jit_index});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_index = at::clone(index);
  auto trt_params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {trt_index});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, trt_params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenIndexSelectNegativeDimConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor, %index : Int (5)):
        %2 : int = prim::Constant[value=-1]()
        %3 : Tensor = aten::index_select(%0, %2, %index)
        return (%3))IR";
  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {5, 3, 9}, {at::kCUDA});
  auto index = at::randint(0, 9, {5}, {at::kCUDA}).to(torch::kI32);

  auto jit_in = at::clone(in);
  auto jit_index = at::clone(index);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {jit_index});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_index = at::clone(index);
  auto trt_params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {trt_index});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, trt_params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
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

TEST(Converters, ATenIndexTensorRepeatedFullIndicesConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor,
            %index0 : Tensor,
            %index1 : Tensor,
            %index2 : Tensor):
        %18 : Tensor?[] = prim::ListConstruct(%index0, %index1, %index2)
        %19 : Tensor = aten::index(%x.1, %18)
        %20 : Tensor = aten::index(%x.1, %18)
        return (%19, %20))IR";

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

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[1], trt_results[1], 2e-6));
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

TEST(Converters, ATenIndexTensorNoneIdx1ConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor,
            %index0 : Tensor):
        %5 : NoneType = prim::Constant()
        %18 : Tensor?[] = prim::ListConstruct(%5, %index0)
        %19 : Tensor = aten::index(%x.1, %18)
        return (%19))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in1 = at::randint(1, 10, {1, 3, 480, 928}, {at::kCUDA});
  auto index0 = at::tensor({2, 1, 0}, {at::kCUDA}).to(torch::kLong);

  auto index0_trt = index0.to(torch::kInt32);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in1, index0});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in1, index0_trt});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}