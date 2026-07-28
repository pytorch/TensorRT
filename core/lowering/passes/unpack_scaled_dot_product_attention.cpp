#include "torch/csrc/jit/ir/subgraph_matcher.h"
#include "torch/csrc/jit/passes/subgraph_rewrite.h"

#include "core/util/prelude.h"
#include "torch/csrc/jit/ir/irparser.h"

namespace torch_tensorrt {
namespace core {
namespace lowering {
namespace passes {

// https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
void UnpackScaledDotProductAttention(std::shared_ptr<torch::jit::Graph>& graph) {
  std::string sdpa_pattern = R"IR(
    graph(%query, %key, %value, %attn_mask, %dropout_p, %is_causal, %scale, %enable_gqa):
      %out: Tensor = aten::scaled_dot_product_attention(%query, %key, %value, %attn_mask, %dropout_p, %is_causal, %scale, %enable_gqa)
      return (%out))IR";

  std::string unpacked_sdpa_pattern = R"IR(
    graph(%query, %key, %value, %attn_mask, %dropout_p, %is_causal, %scale, %enable_gqa):
      %none : NoneType = prim::Constant()
      %1 : int = prim::Constant[value=-1]()
      %2 : int = prim::Constant[value=-2]()
      %3 : int = aten::size(%query, %1)
      %q_size : Long() = prim::NumToTensor(%3)
      %sqrt : Tensor = aten::sqrt(%q_size)
      %scale_factor : Tensor = aten::reciprocal(%sqrt)
      %key_transpose : Tensor = aten::transpose(%key, %2, %1)
      %matmul : Tensor = aten::matmul(%query, %key_transpose)
      %attn_weight : Tensor = aten::mul(%matmul, %scale_factor)
      %softmax : Tensor = aten::softmax(%attn_weight, %1, %none)
      %out : Tensor = aten::matmul(%softmax, %value)
      return(%out))IR";

  std::string unpacked_sdpa_attn_biased_pattern = R"IR(
    graph(%query, %key, %value, %attn_mask, %dropout_p, %is_causal, %scale, %enable_gqa):
      %none : NoneType = prim::Constant()
      %0 : int = prim::Constant[value=1]()
      %1 : int = prim::Constant[value=-1]()
      %2 : int = prim::Constant[value=-2]()
      %3 : int = aten::size(%query, %1)
      %q_size : Long() = prim::NumToTensor(%3)
      %sqrt : Tensor = aten::sqrt(%q_size)
      %scale_factor : Tensor = aten::reciprocal(%sqrt)
      %key_transpose : Tensor = aten::transpose(%key, %2, %1)
      %matmul : Tensor = aten::matmul(%query, %key_transpose)
      %attn_weight : Tensor = aten::mul(%matmul, %scale_factor)
      %attn_bias : Tensor = trt::attn_bias_from_attn_mask(%attn_mask)
      %attn_weight_with_bias : Tensor = aten::add(%attn_weight, %attn_bias, %0)
      %softmax : Tensor = aten::softmax(%attn_weight_with_bias, %1, %none)
      %out : Tensor = aten::matmul(%softmax, %value)
      return(%out))IR";

  // rewrite with None attn_mask
  torch::jit::SubgraphRewriter sdpa_rewriter;
  sdpa_rewriter.RegisterRewritePattern(sdpa_pattern, unpacked_sdpa_pattern);
  sdpa_rewriter.runOnGraph(
      graph, [](const torch::jit::Match& match, const std::unordered_map<std::string, torch::jit::Value*>&) {
        auto is_causal_node = match.anchor->inputs().at(5)->node();
        if (is_causal_node->kind() != at::prim::Constant) {
          LOG_WARNING("Could not unpack scaled_dot_product_attention with non constant is_causal: " << *is_causal_node);
          return false;
        }
        if (is_causal_node->i(at::attr::value) == 1) {
          LOG_WARNING("Could not unpack scaled_dot_product_attention with is_causal = True: " << *is_causal_node);
          return false;
        }
        auto attn_mask_node = match.anchor->inputs().at(3)->node();
        if (attn_mask_node->kind() != at::prim::Constant || !attn_mask_node->mustBeNone()) {
          return false;
        }
        auto enable_gqa_node = match.anchor->inputs().at(7)->node();
        if (enable_gqa_node->kind() != at::prim::Constant) {
          LOG_WARNING(
              "Could not unpack scaled_dot_product_attention with non constant enable_gqa: " << *enable_gqa_node);
          return false;
        }
        if (enable_gqa_node->i(at::attr::value) == 1) {
          LOG_WARNING("Could not unpack scaled_dot_product_attention with enable_gqa = True: " << *enable_gqa_node);
          return false;
        }
        return true;
      });

  // rewrite with float/bool attn_mask this uses a custom op to implement the divergent behavior between bool and float
  // masks without a conditional
  torch::jit::SubgraphRewriter sdpa_attn_mask_rewriter;
  sdpa_attn_mask_rewriter.RegisterRewritePattern(sdpa_pattern, unpacked_sdpa_attn_biased_pattern);
  sdpa_attn_mask_rewriter.runOnGraph(
      graph, [](const torch::jit::Match& match, const std::unordered_map<std::string, torch::jit::Value*>&) {
        auto is_causal_node = match.anchor->inputs().at(5)->node();
        if (is_causal_node->kind() != at::prim::Constant || is_causal_node->i(at::attr::value) == 1) {
          // messages already written in first pass, do not write again
          return false;
        }
        auto enable_gqa_node = match.anchor->inputs().at(7)->node();
        if (enable_gqa_node->kind() != at::prim::Constant || enable_gqa_node->i(at::attr::value) == 1) {
          // messages already written in first pass, do not write again
          return false;
        }
        return true;
      });
  LOG_GRAPH("Post unpack scaled_dot_product_attention: " << *graph);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
