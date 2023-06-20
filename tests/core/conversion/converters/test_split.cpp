#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

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

TEST(Converters, ATenSplitNegativeDimsConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
          %2 : int = prim::Constant[value=1]()
          %n1 : int = prim::Constant[value=-1]()
          %3 : Tensor[] = aten::split(%x.1, %2, %n1)
          %4 : Tensor, %5 : Tensor, %6 : Tensor, %7 : Tensor = prim::ListUnpack(%3)
          return (%4, %5, %6, %7))IR";

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