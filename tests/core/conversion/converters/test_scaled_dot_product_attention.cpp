#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ATenScaledDotProductAttentionConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%query : Tensor, %key : Tensor, %value : Tensor):
        %none : NoneType = prim::Constant()
        %0 : float = prim::Constant[value=0.]()
        %scale : NoneType = prim::Constant()
        %enable_gqa : bool = prim::Constant[value=0]()
        %false : bool = prim::Constant[value=0]()
        %3 : Tensor = aten::scaled_dot_product_attention(%query, %key, %value, %none, %0, %false, %scale, %enable_gqa)
        return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto query = at::rand({32, 8, 128, 64}, {at::kCUDA});
  auto key = at::rand({32, 8, 128, 64}, {at::kCUDA});
  auto value = at::rand({32, 8, 128, 64}, {at::kCUDA});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {query, key, value});

  torch_tensorrt::core::lowering::passes::UnpackScaledDotProductAttention(g);

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {query, key, value});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0]));
}

TEST(Converters, ATenScaledDotProductAttnMaskFloatConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%query : Tensor, %key : Tensor, %value : Tensor, %attn_mask : Tensor):
        %0 : float = prim::Constant[value=0.]()
        %false : bool = prim::Constant[value=0]()
        %scale : NoneType = prim::Constant()
        %enable_gqa : bool = prim::Constant[value=0]()
        %3 : Tensor = aten::scaled_dot_product_attention(%query, %key, %value, %attn_mask, %0, %false, %scale, %enable_gqa)
        return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto query = at::rand({32, 8, 128, 64}, {at::kCUDA});
  auto key = at::rand({32, 8, 128, 64}, {at::kCUDA});
  auto value = at::rand({32, 8, 128, 64}, {at::kCUDA});
  auto attn_mask = at::rand({32, 8, 128, 128}, {at::kCUDA});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {query, key, value, attn_mask});

  torch_tensorrt::core::lowering::passes::UnpackScaledDotProductAttention(g);

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {query, key, value, attn_mask});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0]));
}

TEST(Converters, ATenScaledDotProductAttnMaskBoolDoesNotProduceNaN) {
  const auto graph = R"IR(
      graph(%query : Tensor, %key : Tensor, %value : Tensor, %attn_mask : Tensor):
        %0 : float = prim::Constant[value=0.]()
        %false : bool = prim::Constant[value=0]()
        %scale : NoneType = prim::Constant()
        %enable_gqa : bool = prim::Constant[value=0]()
        %3 : Tensor = aten::scaled_dot_product_attention(%query, %key, %value, %attn_mask, %0, %false, %scale, %enable_gqa)
        return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto query = at::arange(16, {at::kCUDA}).to(at::kFloat).reshape({1, 1, 4, 4}) / 16.0;
  auto key = at::arange(16, {at::kCUDA}).to(at::kFloat).reshape({1, 1, 4, 4}) / 13.0;
  auto value = at::arange(16, {at::kCUDA}).to(at::kFloat).reshape({1, 1, 4, 4}) / 11.0;
  auto attn_mask =
      at::tensor(
          {1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1}, at::TensorOptions().dtype(at::kBool).device(at::kCUDA))
          .reshape({1, 1, 4, 4});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {query, key, value, attn_mask});

  torch_tensorrt::core::lowering::passes::UnpackScaledDotProductAttention(g);

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {query, key, value, attn_mask});

  ASSERT_FALSE(at::isnan(jit_results[0]).any().item<bool>());
  ASSERT_FALSE(at::isnan(trt_results[0]).any().item<bool>());
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0]));
}

TEST(Converters, ATenScaledDotProductAttnMaskIntConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%query : Tensor, %key : Tensor, %value : Tensor, %attn_mask : Tensor):
        %0 : float = prim::Constant[value=0.]()
        %false : bool = prim::Constant[value=0]()
        %scale : NoneType = prim::Constant()
        %enable_gqa : bool = prim::Constant[value=0]()
        %3 : Tensor = aten::scaled_dot_product_attention(%query, %key, %value, %attn_mask, %0, %false, %scale, %enable_gqa)
        return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto query = at::rand({32, 8, 128, 64}, {at::kCUDA});
  auto key = at::rand({32, 8, 128, 64}, {at::kCUDA});
  auto value = at::rand({32, 8, 128, 64}, {at::kCUDA});
  auto attn_mask = at::randint(0, 2, {32, 8, 128, 128}, {at::kCUDA});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {query, key, value, attn_mask});

  torch_tensorrt::core::lowering::passes::UnpackScaledDotProductAttention(g);

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {query, key, value, attn_mask});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0]));
}
