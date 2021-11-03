#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

#ifndef DISABLE_TEST_IN_CI

TEST(Converters, ATenGRUCellConvertsCorrectlyWithBiasCheckHidden) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Tensor,
            %2 : Tensor,
            %3 : Tensor,
            %4 : Tensor,
            %5 : Tensor):
        %6 : Tensor = aten::gru_cell(%0, %1, %2, %3, %4, %5)
        return (%6))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto input = at::randn({50, 10}, {at::kCUDA});
  auto h0 = at::randn({50, 20}, {at::kCUDA});
  auto w_ih = at::randn({3 * 20, 10}, {at::kCUDA});
  auto w_hh = at::randn({3 * 20, 20}, {at::kCUDA});
  auto b_ih = at::randn({3 * 20}, {at::kCUDA});
  auto b_hh = at::randn({3 * 20}, {at::kCUDA});

  auto jit_input = at::clone(input);
  auto jit_h0 = at::clone(h0);
  auto jit_w_ih = at::clone(w_ih);
  auto jit_w_hh = at::clone(w_hh);
  auto jit_b_ih = at::clone(b_ih);
  auto jit_b_hh = at::clone(b_hh);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results =
      torch_tensorrt::tests::util::RunGraph(g, params, {jit_input, jit_h0, jit_w_ih, jit_w_hh, jit_b_ih, jit_b_hh});

  auto trt_input = at::clone(input);
  auto trt_h0 = at::clone(h0);
  auto trt_w_ih = at::clone(w_ih);
  auto trt_w_hh = at::clone(w_hh);
  auto trt_b_ih = at::clone(b_ih);
  auto trt_b_hh = at::clone(b_hh);

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(
      g, params, {trt_input, trt_h0, trt_w_ih, trt_w_hh, trt_b_ih, trt_b_hh});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-5));
}

TEST(Converters, ATenGRUCellConvertsCorrectlyWithoutBiasCheckHidden) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Tensor,
            %2 : Tensor,
            %3 : Tensor):
        %4 : None = prim::Constant()
        %5 : None = prim::Constant()
        %6 : Tensor = aten::gru_cell(%0, %1, %2, %3, %4, %5)
        return (%6))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto input = at::randn({50, 10}, {at::kCUDA});
  auto h0 = at::randn({50, 20}, {at::kCUDA});
  auto w_ih = at::randn({3 * 20, 10}, {at::kCUDA});
  auto w_hh = at::randn({3 * 20, 20}, {at::kCUDA});

  auto jit_input = at::clone(input);
  auto jit_h0 = at::clone(h0);
  auto jit_w_ih = at::clone(w_ih);
  auto jit_w_hh = at::clone(w_hh);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_input, jit_h0, jit_w_ih, jit_w_hh});

  auto trt_input = at::clone(input);
  auto trt_h0 = at::clone(h0);
  auto trt_w_ih = at::clone(w_ih);
  auto trt_w_hh = at::clone(w_hh);

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_input, trt_h0, trt_w_ih, trt_w_hh});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-5));
}

TEST(Converters, ATenLSTMCellConvertsCorrectlyWithBiasCheckHidden) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Tensor,
            %2 : Tensor,
            %3 : Tensor,
            %4 : Tensor,
            %5 : Tensor,
            %6 : Tensor):
        %7 : Tensor[] = prim::ListConstruct(%1, %2)
        %8 : Tensor, %9 : Tensor = aten::lstm_cell(%0, %7, %3, %4, %5, %6)
        return (%8))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto input = at::randn({50, 10}, {at::kCUDA});
  auto h0 = at::randn({50, 20}, {at::kCUDA});
  auto c0 = at::randn({50, 20}, {at::kCUDA});
  auto w_ih = at::randn({4 * 20, 10}, {at::kCUDA});
  auto w_hh = at::randn({4 * 20, 20}, {at::kCUDA});
  auto b_ih = at::randn({4 * 20}, {at::kCUDA});
  auto b_hh = at::randn({4 * 20}, {at::kCUDA});

  auto jit_input = at::clone(input);
  auto jit_h0 = at::clone(h0);
  auto jit_c0 = at::clone(c0);
  auto jit_w_ih = at::clone(w_ih);
  auto jit_w_hh = at::clone(w_hh);
  auto jit_b_ih = at::clone(b_ih);
  auto jit_b_hh = at::clone(b_hh);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(
      g, params, {jit_input, jit_h0, jit_c0, jit_w_ih, jit_w_hh, jit_b_ih, jit_b_hh});

  auto trt_input = at::clone(input);
  auto trt_h0 = at::clone(h0);
  auto trt_c0 = at::clone(c0);
  auto trt_w_ih = at::clone(w_ih);
  auto trt_w_hh = at::clone(w_hh);
  auto trt_b_ih = at::clone(b_ih);
  auto trt_b_hh = at::clone(b_hh);

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(
      g, params, {trt_input, trt_h0, trt_c0, trt_w_ih, trt_w_hh, trt_b_ih, trt_b_hh});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-5));
}

TEST(Converters, ATenLSTMCellConvertsCorrectlyWithBiasCheckCell) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Tensor,
            %2 : Tensor,
            %3 : Tensor,
            %4 : Tensor,
            %5 : Tensor,
            %6 : Tensor):
        %7 : Tensor[] = prim::ListConstruct(%1, %2)
        %8 : Tensor, %9 : Tensor = aten::lstm_cell(%0, %7, %3, %4, %5, %6)
        return (%9))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto input = at::randn({50, 10}, {at::kCUDA});
  auto h0 = at::randn({50, 20}, {at::kCUDA});
  auto c0 = at::randn({50, 20}, {at::kCUDA});
  auto w_ih = at::randn({4 * 20, 10}, {at::kCUDA});
  auto w_hh = at::randn({4 * 20, 20}, {at::kCUDA});
  auto b_ih = at::randn({4 * 20}, {at::kCUDA});
  auto b_hh = at::randn({4 * 20}, {at::kCUDA});

  auto jit_input = at::clone(input);
  auto jit_h0 = at::clone(h0);
  auto jit_c0 = at::clone(c0);
  auto jit_w_ih = at::clone(w_ih);
  auto jit_w_hh = at::clone(w_hh);
  auto jit_b_ih = at::clone(b_ih);
  auto jit_b_hh = at::clone(b_hh);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(
      g, params, {jit_input, jit_h0, jit_c0, jit_w_ih, jit_w_hh, jit_b_ih, jit_b_hh});

  auto trt_input = at::clone(input);
  auto trt_h0 = at::clone(h0);
  auto trt_c0 = at::clone(c0);
  auto trt_w_ih = at::clone(w_ih);
  auto trt_w_hh = at::clone(w_hh);
  auto trt_b_ih = at::clone(b_ih);
  auto trt_b_hh = at::clone(b_hh);

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(
      g, params, {trt_input, trt_h0, trt_c0, trt_w_ih, trt_w_hh, trt_b_ih, trt_b_hh});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-5));
}

TEST(Converters, ATenLSTMCellConvertsCorrectlyWithoutBiasCheckHidden) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Tensor,
            %2 : Tensor,
            %3 : Tensor,
            %4 : Tensor):
        %5 : None = prim::Constant()
        %6 : None = prim::Constant()
        %7 : Tensor[] = prim::ListConstruct(%1, %2)
        %8 : Tensor, %9 : Tensor = aten::lstm_cell(%0, %7, %3, %4, %5, %6)
        return (%8))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto input = at::randn({50, 10}, {at::kCUDA});
  auto h0 = at::randn({50, 20}, {at::kCUDA});
  auto c0 = at::randn({50, 20}, {at::kCUDA});
  auto w_ih = at::randn({4 * 20, 10}, {at::kCUDA});
  auto w_hh = at::randn({4 * 20, 20}, {at::kCUDA});

  auto jit_input = at::clone(input);
  auto jit_h0 = at::clone(h0);
  auto jit_c0 = at::clone(c0);
  auto jit_w_ih = at::clone(w_ih);
  auto jit_w_hh = at::clone(w_hh);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_input, jit_h0, jit_c0, jit_w_ih, jit_w_hh});

  auto trt_input = at::clone(input);
  auto trt_h0 = at::clone(h0);
  auto trt_c0 = at::clone(c0);
  auto trt_w_ih = at::clone(w_ih);
  auto trt_w_hh = at::clone(w_hh);

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results =
      torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_input, trt_h0, trt_c0, trt_w_ih, trt_w_hh});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-5));
}

TEST(Converters, ATenLSTMCellConvertsCorrectlyWithoutBiasCheckCell) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Tensor,
            %2 : Tensor,
            %3 : Tensor,
            %4 : Tensor):
        %5 : None = prim::Constant()
        %6 : None = prim::Constant()
        %7 : Tensor[] = prim::ListConstruct(%1, %2)
        %8 : Tensor, %9 : Tensor = aten::lstm_cell(%0, %7, %3, %4, %5, %6)
        return (%9))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto input = at::randn({50, 10}, {at::kCUDA});
  auto h0 = at::randn({50, 20}, {at::kCUDA});
  auto c0 = at::randn({50, 20}, {at::kCUDA});
  auto w_ih = at::randn({4 * 20, 10}, {at::kCUDA});
  auto w_hh = at::randn({4 * 20, 20}, {at::kCUDA});

  auto jit_input = at::clone(input);
  auto jit_h0 = at::clone(h0);
  auto jit_c0 = at::clone(c0);
  auto jit_w_ih = at::clone(w_ih);
  auto jit_w_hh = at::clone(w_hh);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_input, jit_h0, jit_c0, jit_w_ih, jit_w_hh});

  auto trt_input = at::clone(input);
  auto trt_h0 = at::clone(h0);
  auto trt_c0 = at::clone(c0);
  auto trt_w_ih = at::clone(w_ih);
  auto trt_w_hh = at::clone(w_hh);

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results =
      torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_input, trt_h0, trt_c0, trt_w_ih, trt_w_hh});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-5));
}

#endif
