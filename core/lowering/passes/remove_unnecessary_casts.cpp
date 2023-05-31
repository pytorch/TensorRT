#include "torch/csrc/jit/ir/constants.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/subgraph_rewrite.h"

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
                if (potential_cast->kind() == c10::Symbol::fromQualString("aten::Int") ||
                    potential_cast->kind() == c10::Symbol::fromQualString("aten::Float")) {
                  LOG_GRAPH("Downstream user is aten::Int/aten::Float");
                  auto arg = use.offset;

                  for (size_t k = 0; k < user->inputs().size(); ++k) {
                    if (k != arg) {
                      if (user->inputs()[k]->type()->isSubtypeOf(c10::TensorType::get())) {
                        LOG_GRAPH("Input " << k << " is a Tensor");
                        if (user->inputs()[k]->node()->kind() == c10::Symbol::fromQualString("prim::NumToTensor")) {
                          auto num_to_tensor = user->inputs()[k]->node();

                          LOG_GRAPH(
                              "Found a prim::NumToTensor / aten::[Int/Float] pair with an intermediate operation:\n    "
                              << *(*it) << *num_to_tensor << *user << *potential_cast);

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
                            case c10::aten::floor_divide:
                              new_node = g->create(c10::aten::floordiv, user->inputs(), 1);
                              new_node->insertAfter(user);
                              new_node->outputs()[0]->setType(c10::IntType::get());
                              user->outputs()[0]->replaceAllUsesWith(new_node->outputs()[0]);
                              user->destroy();
                              break;
                            case c10::aten::div:
                              // If the first two entries to aten::div are non-Tensors,
                              // there cannot be a rounding mode specified (3rd entry)
                              if (!user->inputs()[0]->type()->isSubtypeOf(c10::TensorType::get()) &&
                                  !user->inputs()[1]->type()->isSubtypeOf(c10::TensorType::get()) &&
                                  user->inputs().size() == 3 &&
                                  user->inputs()[2]->type()->isSubtypeOf(c10::StringType::get()) &&
                                  torch::jit::toIValue(user->inputs()[2]).has_value()) {
                                // Select the first 2 entries of the inputs, corresponding to the values
                                auto div_args = user->inputs().slice(0, 2);

                                // Depending on the rounding mode, create the appropriate nodes
                                if (torch::jit::toIValue(user->inputs()[2]).value().toStringRef() == "trunc") {
                                  // Truncate case (round result towards 0)
                                  torch::jit::Node* new_node_div;
                                  // Create node which simply divides the two entries
                                  new_node_div = g->create(c10::aten::div, div_args, 1);
                                  new_node_div->insertAfter(user);
                                  new_node_div->outputs()[0]->setType(c10::FloatType::get());

                                  // Create node which casts the result to an integer, effectively truncating
                                  new_node = g->create(c10::aten::Int, new_node_div->outputs(), 1);
                                  new_node->insertAfter(new_node_div);
                                  new_node->outputs()[0]->setType(c10::IntType::get());

                                  user->outputs()[0]->replaceAllUsesWith(new_node->outputs()[0]);
                                  user->destroy();
                                  break;

                                } else if (torch::jit::toIValue(user->inputs()[2]).value().toStringRef() == "floor") {
                                  // Floor case (round result down)
                                  // Replace aten::div with aten::floordiv
                                  new_node = g->create(c10::aten::floordiv, div_args, 1);
                                  new_node->insertAfter(user);
                                  new_node->outputs()[0]->setType(c10::IntType::get());

                                  user->outputs()[0]->replaceAllUsesWith(new_node->outputs()[0]);
                                  user->destroy();
                                  break;
                                }
                              }

                            default:
                              new_node = g->create(user->kind(), user->inputs(), 1);
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
  LOG_GRAPH("Post removing single use 0-dim Tensor operations: " << *g);
}

// Schemas for Aten::Int which can be replaced by scalar equivalents
const std::unordered_set<c10::Symbol> AtenIntReplacementNodeKinds = {
    torch::jit::aten::mul,
    torch::jit::aten::floor_divide,
};

c10::optional<torch::jit::Value*> Validate0DTensor(torch::jit::Value* value) {
  // Validates that the input Value* is a 0D Tensor (or int/float)
  // Return the stored int/float Value* if so, otherwise null
  c10::optional<torch::jit::Value*> enclosed_scalar_value = {};

  // Regular Int/Float case
  if (value->type()->isSubtypeOf(c10::IntType::get()) || value->type()->isSubtypeOf(c10::FloatType::get())) {
    enclosed_scalar_value = value;
    return enclosed_scalar_value;
  }

  // Constant Tensor case
  if (value->node()->kind() == torch::jit::prim::Constant && value->type()->isSubtypeOf(c10::TensorType::get())) {
    // Retrieve the Tensor stored in constant
    at::Tensor t = *torch::jit::constant_as<at::Tensor>(value);
    // Validate the shape of the Tensor is 0D (single-element) and integral
    if (t.sizes() == std::vector<int64_t>({}) && t.item().isIntegral(false)) {
      // Extract the stored value, add it to the graph as a constant
      torch::jit::WithInsertPoint guard(value->node());
      auto new_const_val = value->owningGraph()->insertConstant(t.item(), c10::nullopt, value->node()->scope());
      new_const_val->copyMetadata(value);
      new_const_val->setType(c10::IntType::get());
      enclosed_scalar_value = new_const_val;
      return enclosed_scalar_value;
    } else {
      LOG_DEBUG("In aten::Int.Tensor removal, encountered a const which was either not 0D or not integral");
    }
  }

  // NumToTensor case
  if (value->node()->kind() == torch::jit::prim::NumToTensor && value->type()->isSubtypeOf(c10::TensorType::get())) {
    // Input to NumToTensor is relevant scalar
    enclosed_scalar_value = value->node()->input();
    return enclosed_scalar_value;
  }

  return enclosed_scalar_value;
}

c10::optional<torch::jit::Value*> TracebackAndEliminate0DTensors(torch::jit::Node* node) {
  // Trace back through a node and all parents to eliminate 0D Tensors
  // and update schemas to their scalar alternatives, returning final
  // Value* to user

  // Requires valid schema with at least two inputs
  if (AtenIntReplacementNodeKinds.find(node->kind()) == AtenIntReplacementNodeKinds.end() ||
      node->inputs().size() < 2) {
    LOG_DEBUG(
        "Encountered node " << node->kind().toQualString()
                            << " which is unsupported in the aten::Int.Tensor replacement lowering pass.");
    return {};
  }

  // Validate the first and second function inputs are 0D tensors or scalars
  c10::optional<torch::jit::Value*> first_input_scalar_value = Validate0DTensor(node->inputs()[0]);
  c10::optional<torch::jit::Value*> second_input_scalar_value = Validate0DTensor(node->inputs()[1]);

  // If the first input is not a scalar, recursively traceback on parent nodes
  if (!first_input_scalar_value.has_value()) {
    LOG_DEBUG("In aten::Int.Tensor lowering, now tracing " << node->inputs()[0]->node()->kind().toQualString());
    first_input_scalar_value = TracebackAndEliminate0DTensors(node->inputs()[0]->node());
  }

  // If the second input is not a scalar, recursively traceback on parent nodes
  if (!second_input_scalar_value.has_value()) {
    LOG_DEBUG("In aten::Int.Tensor lowering, now tracing " << node->inputs()[0]->node()->kind().toQualString());
    second_input_scalar_value = TracebackAndEliminate0DTensors(node->inputs()[1]->node());
  }

  if (!first_input_scalar_value.has_value() || !second_input_scalar_value.has_value()) {
    LOG_DEBUG(
        "In aten::Int.Tensor lowering, recursive trace through node input "
        << "parents failed to return a Scalar value for at least one parent node.");
    return {};
  }

  // Set default insert point at node
  torch::jit::WithInsertPoint guard(node);
  torch::jit::Node* new_node;

  switch (node->kind()) {
    // In the aten::floor_divide case, the schema syntax changes, so a new node
    // must be inserted
    case torch::jit::aten::floor_divide:
      new_node = node->owningGraph()->create(
          torch::jit::aten::floordiv, {first_input_scalar_value.value(), second_input_scalar_value.value()}, 1);
      new_node->insertAfter(node);
      new_node->output()->setType(c10::IntType::get());
      return new_node->output();

    // In the aten::mul case, the schema syntax is the same, so we can use the existing schema
    // with new inputs
    default:
      new_node = node->owningGraph()->create(
          node->kind(), {first_input_scalar_value.value(), second_input_scalar_value.value()}, 1);
      new_node->insertAfter(node);
      new_node->output()->setType(c10::IntType::get());
      return new_node->output();
  }
}

void ReplaceAtenInt(std::shared_ptr<torch::jit::Graph>& g) {
  // Find all nodes with the aten::Int.Tensor schema and replace those
  // by tracing through the node and resolving the use of 0D tensors
  // to their corresponding scalar alternatives

  // Iterate over all nodes in the graph
  for (auto it = g->block()->nodes().begin(), end = g->block()->nodes().end(); it != end; ++it) {
    // Validate schema requirements for aten::Int.Tensor
    if (it->kind() == torch::jit::aten::Int && it->inputs().size() == 1 &&
        it->input()->type()->isSubtypeOf(c10::TensorType::get())) {
      LOG_DEBUG("Found an aten::Int.Tensor case, attempting to resolve input scalars.");

      // If the node parent schema is of a supported type, trace back through the graph
      if (AtenIntReplacementNodeKinds.find(it->input()->node()->kind()) != AtenIntReplacementNodeKinds.end()) {
        LOG_DEBUG(
            "Tracing parent node " << it->input()->node()->kind().toQualString()
                                   << " to eliminate 0D Tensors for aten::Int.Tensor case.");
        auto scalar_input_value = TracebackAndEliminate0DTensors(it->input()->node());
        if (scalar_input_value.has_value()) {
          it->output()->replaceAllUsesWith(scalar_input_value.value());
          LOG_DEBUG("Tracing parent nodes for aten::Int.Tensor case succeeded.");
        } else {
          LOG_DEBUG("Tracing parent nodes for aten::Int.Tensor case failed.");
        }
      } else {
        LOG_DEBUG(
            "Parent node schema " << it->input()->node()->kind().toQualString()
                                  << " is currently unsupported for aten::Int.Tensor case.");
      }
    }
  }

  // Clean up remnant operators in graph
  torch::jit::EliminateDeadCode(g);
  LOG_GRAPH("Post removing aten.Int.Tensor operations: " << *g);
}

void RemoveCollectionCast(std::shared_ptr<torch::jit::Graph>& g) {
  // Removes unnecessary collection-casting of graph outputs
  // Only to be used if the overall output is intended to be a TRT Engine
  // Will cause errors if used directly as a TorchScript graph

  // Validate the output is a single value with type Tuple or List
  if (!(g->outputs().size() == 1 &&
        (g->outputs()[0]->node()->kind() == torch::jit::prim::TupleConstruct ||
         g->outputs()[0]->node()->kind() == torch::jit::prim::ListConstruct))) {
    return;
  }

  // Ensure all inputs to the Tuple/List Construct operator are regular Tensors
  // (nested structures cannot be preserved in TensorRT)
  auto all_tensors = true;
  auto collection_inputs = g->outputs()[0]->node()->inputs();

  for (size_t i = 0; i < collection_inputs.size(); ++i) {
    all_tensors &= collection_inputs[i]->type()->isSubtypeOf(c10::TensorType::get());
  }

  if (!all_tensors) {
    return;
  }

  // For each input to the collection packing operator, add its value directly
  // as an output of the graph
  for (size_t i = 0; i < collection_inputs.size(); ++i) {
    g->registerOutput(collection_inputs[i]);
  }

  // Remove the original output value of the graph (the collection object)
  g->eraseOutput(0);

  // Clean up remnant collection node in graph
  torch::jit::EliminateDeadCode(g);
  LOG_GRAPH("Post removing collection casting operations: " << *g);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
