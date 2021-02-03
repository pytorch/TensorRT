#include <torch/torch.h>
#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ATenExpandSameDimConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
            %2 : int[] = prim::Constant[value=[3, 4]]()
            %3 : bool = prim::Constant[value=0]()
            %4 : Tensor = aten::expand(%x.1, %2, %3)
            return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {3, 1}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenExpandSameDimConvertsCorrectlyWithDynamicInput) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
            %2 : int[] = prim::Constant[value=[3, 4]]()
            %3 : bool = prim::Constant[value=0]()
            %4 : Tensor = aten::expand(%x.1, %2, %3)
            return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {3, 1}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngineDynamic(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenExpandTileConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
            %2 : int[] = prim::Constant[value=[2, 3, 1]]()
            %3 : bool = prim::Constant[value=0]()
            %4 : Tensor = aten::expand(%x.1, %2, %3)
            return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {3, 1}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenExpandTileConvertsCorrectlyWithDynamicInput) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
            %2 : int[] = prim::Constant[value=[2, 3, 1]]()
            %3 : bool = prim::Constant[value=0]()
            %4 : Tensor = aten::expand(%x.1, %2, %3)
            return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {3, 1}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngineDynamic(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenExpandTileLastConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
            %2 : int[] = prim::Constant[value=[1, 3, 4]]()
            %3 : bool = prim::Constant[value=0]()
            %4 : Tensor = aten::expand(%x.1, %2, %3)
            return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {3, 1}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenExpandTileLastConvertsCorrectlyWithDynamicInput) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
            %2 : int[] = prim::Constant[value=[1, 3, 4]]()
            %3 : bool = prim::Constant[value=0]()
            %4 : Tensor = aten::expand(%x.1, %2, %3)
            return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {3, 1}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngineDynamic(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenExpandNegativeSizeConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
            %2 : int[] = prim::Constant[value=[3, -1, 4]]()
            %3 : bool = prim::Constant[value=0]()
            %4 : Tensor = aten::expand(%x.1, %2, %3)
            return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {3, 1}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenExpandNegativeSizeConvertsCorrectlyWithDynamicInput) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
            %2 : int[] = prim::Constant[value=[3, -1, 4]]()
            %3 : bool = prim::Constant[value=0]()
            %4 : Tensor = aten::expand(%x.1, %2, %3)
            return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {3, 1}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngineDynamic(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

/* Expand_as layer takes two inputs and only dimensions of second input are
   actually used. TRT prunes away the second input. This will result in internal
   failure from TRT. To avoid unrelated issues, we add a dummy operation which
   outputs second_input+2 as a second output. The second input is preserved.
*/
TEST(Converters, ATenExpandASConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor,
      %y.1 : Tensor):
        %3 : int = prim::Constant[value=1]()
        %4 : int = prim::Constant[value=2]()
        %5 : Tensor = aten::expand_as(%x.1, %y.1)
        %6 : Tensor = aten::add(%y.1, %4, %3)
        return (%5, %6))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {3, 1}, {at::kCUDA});
  auto target_in = at::randint(1, 10, {2, 3, 1}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto jit_target_in = at::clone(target_in);
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in, jit_target_in});

  auto trt_in = at::clone(jit_in);
  auto trt_target_in = at::clone(jit_target_in);
  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in, trt_target_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenExpandAsConvertsCorrectlyWithDynamicInput) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor,
      %y.1 : Tensor):
        %3 : int = prim::Constant[value=1]()
        %4 : int = prim::Constant[value=2]()
        %5 : Tensor = aten::expand_as(%x.1, %y.1)
        %6 : Tensor = aten::add(%y.1, %4, %3)
        return (%5, %6))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {3, 1}, {at::kCUDA});
  auto target_in = at::randint(1, 10, {3, 4}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto jit_target_in = at::clone(target_in);
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in, jit_target_in});

  auto trt_in = at::clone(jit_in);
  auto trt_target_in = at::clone(jit_target_in);
  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngineDynamic(g, params, {trt_in, trt_target_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenRepeatConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
            %2 : int[] = prim::Constant[value=[4, 1]]()
            %3 : Tensor = aten::repeat(%x.1, %2)
            return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {1, 3}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(jit_in);
  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenRepeatConvertsCorrectlyWithDynamicInput) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
            %2 : int[] = prim::Constant[value=[4, 1]]()
            %3 : Tensor = aten::repeat(%x.1, %2)
            return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {1, 3}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(jit_in);
  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngineDynamic(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenRepeat3dConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
            %2 : int[] = prim::Constant[value=[2, 2, 2]]()
            %3 : Tensor = aten::repeat(%x.1, %2)
            return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {2, 3, 2}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(jit_in);
  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenRepeat3dConvertsCorrectlyWithDynamicInput) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
            %2 : int[] = prim::Constant[value=[2, 2, 2]]()
            %3 : Tensor = aten::repeat(%x.1, %2)
            return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {2, 3, 2}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(jit_in);
  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngineDynamic(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenRepeatExtraDimsConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
            %2 : int[] = prim::Constant[value=[1, 3, 2]]()
            %3 : Tensor = aten::repeat(%x.1, %2)
            return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {1, 3}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(jit_in);
  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenRepeatExtraDimsConvertsCorrectlyWithDynamicInput) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
            %2 : int[] = prim::Constant[value=[1, 3, 2]]()
            %3 : Tensor = aten::repeat(%x.1, %2)
            return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {1, 3}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(jit_in);
  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngineDynamic(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}