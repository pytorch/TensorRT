#include "torch/csrc/jit/passes/subgraph_rewrite.h"
#include "torch/csrc/jit/ir/constants.h"

#include "core/util/prelude.h"

#include <vector>

namespace torch_tensorrt {
namespace core {
namespace lowering {
namespace passes {


// Presumably this is safe since torch::jit::EraseNumberTypesOnBlock exists which just
// removes prim::TensorToNum, aten::Float, aten::Int and prim::NumToTensor nodes outright
void RemoveUnnecessaryCasts(std::shared_ptr<torch::jit::Graph>& graph) {
  std::string int_cast_pattern = R"IR(
    graph(%1: int):
      %2: Tensor = aten::NumToTensor(%1)
      %3: int = aten::Int(%2)
      return (%3))IR";
  std::string int_clean_pattern = R"IR(
    graph(%1: int):
      return (%1))IR";

  std::string float_cast_pattern = R"IR(
    graph(%1: float):
      %2: Tensor = aten::NumToTensor(%1)
      %3: float = aten::Float(%2)
      return (%3))IR";
  std::string float_clean_pattern = R"IR(
    graph(%1: float):
      return (%1))IR";

  std::string bool_cast_pattern = R"IR(
    graph(%1: bool):
      %2: Tensor = aten::NumToTensor(%1)
      %3: bool = aten::Bool(%2)
      return (%3))IR";
  std::string bool_clean_pattern = R"IR(
    graph(%1: bool):
      return (%1))IR";

  torch::jit::SubgraphRewriter int_cast_rewriter;
  int_cast_rewriter.RegisterRewritePattern(int_cast_pattern, int_clean_pattern);
  int_cast_rewriter.runOnGraph(graph);

  torch::jit::SubgraphRewriter float_cast_rewriter;
  float_cast_rewriter.RegisterRewritePattern(float_cast_pattern, float_clean_pattern);
  float_cast_rewriter.runOnGraph(graph);

  torch::jit::SubgraphRewriter bool_cast_rewriter;
  bool_cast_rewriter.RegisterRewritePattern(bool_cast_pattern, bool_clean_pattern);
  bool_cast_rewriter.runOnGraph(graph);

  LOG_GRAPH("After RemoveUnnecessaryCasts: " << *graph);
}

void RemoveSingleUse0DTensors(std::shared_ptr<torch::jit::Graph>& g) {
  for (auto it = g->block()->nodes().begin(), end = g->block()->nodes().end(); it != end; ++it) {
    if (it->kind() == torch::jit::prim::Constant) {
      // Going from a constant and is single use means we can fuse
      if (it->output()->type()->isSubtypeOf(c10::TensorType::get())) {
        // Get the tensor stored in constant
        at::Tensor t = *torch::jit::constant_as<at::Tensor>(it->output());
        // If shape is 0D
        if (t.sizes() == std::vector<int64_t>({})) {
          LOG_GRAPH("Found a 0D Tensor: " << it->output()->debugName());
          LOG_GRAPH("Number of uses: " << it->output()->uses().size());
          // If the tensor is only used once
          if (it->output()->uses().size() == 1) {
            auto use = it->output()->uses()[0];
            auto user = use.user;

            // Is a NumToTensor / aten::[Int/Float] case
            if (user->outputs().size() == 1 && user->outputs()[0]->type()->isSubtypeOf(c10::TensorType::get())) {
              if (user->output()->uses().size() == 1) {
                auto potential_cast = user->output()->uses()[0].user;
                // The downstream user is aten::Int
                if (potential_cast->kind() == c10::Symbol::fromQualString("aten::Int")
                    || potential_cast->kind() == c10::Symbol::fromQualString("aten::Float")) {
                  LOG_GRAPH("Downstream user is aten::Int/aten::Float");
                  auto arg = use.offset;

                  for (size_t k = 0; k < user->inputs().size(); ++k) {
                    if (k != arg) {
                      if (user->inputs()[k]->type()->isSubtypeOf(c10::TensorType::get())) {
                        LOG_GRAPH("Input " << k << " is a Tensor");
                        if (user->inputs()[k]->node()->kind() == c10::Symbol::fromQualString("prim::NumToTensor")) {
                          auto num_to_tensor = user->inputs()[k]->node();
                          
                          LOG_GRAPH("Found a prim::NumToTensor / aten::[Int/Float] pair with an intermediate operation:\n    " 
                                      << *(*it)
                                      << *num_to_tensor
                                      << *user 
                                      << *potential_cast);
                          
                          // Replace the Tensor Constant with a scalar constant
                          LOG_GRAPH("Deleting 0-dim Tensor: " << **it);
                          torch::jit::WithInsertPoint gaurd(*it);

                          auto new_const_val = g->insertConstant(t.item(), c10::nullopt, it->scope());
                          new_const_val->copyMetadata(it->output());
                          // How to determine the internal scalar type instead of assuming?
                          if (potential_cast->kind() == c10::aten::Int) {
                            new_const_val->setType(c10::IntType::get());
                          } else if (potential_cast->kind() == c10::aten::Float) {
                            new_const_val->setType(c10::FloatType::get());
                          }
                          it->output()->replaceAllUsesWith(new_const_val);
                          it.destroyCurrent();

                          LOG_GRAPH("New constant: " << *new_const_val->node());

                          // Delete NumToTensor
                          LOG_GRAPH("Deleting NumToTensor: " << *num_to_tensor);
                          num_to_tensor->output()->replaceAllUsesWith(num_to_tensor->inputs()[0]);
                          num_to_tensor->destroy();

                          // Change intermediate op output type
                          LOG_GRAPH(user->schema());

                          torch::jit::Node* new_node;
                          switch (user->kind()) {
                            // Use this to handle special cases where the scalar version of the intermediate operator
                            // has a different schema than the original
                            case c10::aten::add:
                              new_node = g->create(
                                user->kind(),
                                torch::jit::ArrayRef<torch::jit::Value*>({user->inputs()[0], user->inputs()[1]}),
                                1);
                              new_node->insertAfter(user);
                              new_node->outputs()[0]->setType(c10::IntType::get());
                              user->outputs()[0]->replaceAllUsesWith(new_node->outputs()[0]);
                              user->destroy();
                              break;
                            default:
                              new_node = g->create(
                                user->kind(),
                                user->inputs(),
                                1);
                              new_node->insertAfter(user);
                              new_node->outputs()[0]->setType(c10::IntType::get());
                              user->outputs()[0]->replaceAllUsesWith(new_node->outputs()[0]);
                              user->destroy();
                              break;
                          }

                          LOG_GRAPH("New intermediate operation: " << *new_node);
                          LOG_GRAPH(new_node->schema());
                          
                          // Delete aten::Int
                          LOG_GRAPH("Deleting aten::[Int/Float]: " << *potential_cast);
                          potential_cast->output()->replaceAllUsesWith(potential_cast->inputs()[0]);
                          potential_cast->destroy();
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    } 
  }
  LOG_ERROR("Post removing single use 0-dim Tensor operations: " << *g);
}


} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
