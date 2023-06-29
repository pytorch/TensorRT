#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/passes/create_functional_graphs.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/erase_number_types.h"
#include "torch/csrc/jit/passes/freeze_module.h"
#include "torch/csrc/jit/passes/fuse_linear.h"
#include "torch/csrc/jit/passes/guard_elimination.h"
#include "torch/csrc/jit/passes/loop_unrolling.h"
#include "torch/csrc/jit/passes/lower_graph.h"
#include "torch/csrc/jit/passes/lower_tuples.h"
#include "torch/csrc/jit/passes/peephole.h"
#include "torch/csrc/jit/passes/remove_exceptions.h"
#include "torch/csrc/jit/passes/remove_mutation.h"

#include "core/lowering/lowering.h"
#include "core/lowering/passes/passes.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace lowering {

void DropUnusedNodes(torch::jit::Block* b);

void LowerBlock(torch::jit::Block* b) {
  DropUnusedNodes(b);
}

int AutocastLongInputs(
    std::shared_ptr<torch::jit::Graph>& g,
    ir::TypeMap input_type_map,
    std::string target_device_name) {
  int num_autocasts = 0;
  auto old_insert_point = g->insertPoint();
  g->setInsertPoint(g->nodes().front());
  // For each graph input, determine if it can be autocasted
  for (size_t i = 0; i < g->inputs().size(); i++) {
    auto input = g->inputs()[i];

    // Autocasted inputs must be Tensor-type
    if (input->type()->isSubtypeOf(c10::TensorType::get())) {
      auto dtype_input = input_type_map.find(input);

      // Ensure the data type to be casted to exists in the type map
      if (dtype_input == input_type_map.end() || !dtype_input->second) {
        LOG_DEBUG("No inferred input dtype for tensor " << input->debugName() << ", skipping autocast");
        continue;
      }

      auto dtype = dtype_input->second.value();
      // Currently, we do not autocast inputs for which the determined type is not long
      if (dtype != at::kLong) {
        LOG_DEBUG(
            "Skipping autocast for tensor " << input->debugName() << ", since its dtype is " << dtype
                                            << " and not at::kLong");
        continue;
      }

      LOG_DEBUG("Inserting aten::to casting " << input->debugName() << " to dtype " << dtype);

      // Generate cast node sending input tensors to the inferred or specified datatype (long)
      torch::jit::Value *const_false, *cuda, *none_val;
      if (num_autocasts == 0) {
        // Only generate constants once and reuse for all autocasts
        const_false = g->insertConstant(0);
        const_false->setType(torch::jit::BoolType::get());
        cuda = g->insertConstant(target_device_name);
        cuda->setType(torch::jit::DeviceObjType::get());
        none_val = g->insertNode(g->createNone())->output();
      }

      auto const_type = g->insertConstant(dtype);
      auto cast_node = g->create(torch::jit::aten::to, {input, cuda, const_type, const_false, const_false, none_val});

      // Replace all uses of the original tensor with that of the casted tensor
      g->insertNode(cast_node);
      input->replaceAllUsesAfterNodeWith(cast_node, cast_node->outputs()[0]);

      // Mark the cast node to run in PyTorch for ease of casting
      LOG_GRAPH("Marking autocast node " << util::node_info(cast_node) << " to run in PyTorch");
      cast_node->i_(c10::Symbol::attr("to_compile"), (int64_t) false);
      num_autocasts++;
    }
  }
  g->setInsertPoint(old_insert_point);
  LOG_GRAPH("Inserted " << num_autocasts << " autocasts");

  if (num_autocasts > 0) {
    LOG_WARNING(
        "Data types for input tensors have been modified by inserting "
        << "aten::to operations which cast INT64 inputs to INT32. "
        << "To disable this, please recompile using INT32 inputs");

    LOG_GRAPH("Graph after Autocast: " << *g);
  }

  return num_autocasts;
}

void LowerGraph(std::shared_ptr<torch::jit::Graph>& g, std::vector<torch::jit::IValue>& params, LowerInfo lower_info) {
  torch::jit::EliminateRedundantGuards(g);
  torch::jit::RemoveListMutation(g);
  torch::jit::RemoveTensorMutation(g);
  torch::jit::CreateFunctionalGraphs(g);
  torch::jit::InlineFunctionalGraphs(g);
  torch::jit::PeepholeOptimize(g, false);
  torch::jit::FuseLinear(g);
  torch::jit::EliminateExceptions(g);
  if (!lower_info.disable_cse) {
    torch::jit::EliminateCommonSubexpression(g);
  }
  torch::jit::EliminateDeadCode(g);
  if (lower_info.forced_fallback_modules.size() > 0) {
    passes::MarkNodesForFallback(g, true);
  }
  passes::UnpackHardSwish(g);
  passes::UnpackHardSigmoid(g);
  passes::EliminateExceptionOrPassPattern(g);
  passes::ReduceToOperation(g);
  passes::ReduceGelu(g);
  passes::ReduceRemainder(g);
  passes::RemoveContiguous(g);
  passes::ViewToReshape(g);
  passes::RemoveDropout(g);
  passes::LinearToAddMM(g);
  passes::Conv1DToConvolution(g);
  passes::ConvTransposed1DToConvolution(g);
  passes::Conv2DToConvolution(g);
  passes::ConvTransposed2DToConvolution(g);
  passes::Conv3DToConvolution(g);
  passes::ConvTransposed3DToConvolution(g);
  passes::FuseAddMMBranches(g);
  passes::RemoveBNDimCheck(g);
  // torch::jit::UnrollLoops(g);
  passes::UnpackAddMM(g);
  // passes::UnpackBatchNorm(g);
  passes::UnpackLogSoftmax(g);
  passes::UnpackRsqrt(g);
  passes::UnpackStd(g);
  passes::UnpackVar(g);
  passes::RemoveNOPs(g);
  passes::AliasOperators(g);
  passes::SiluToSigmoidMultipication(g);
  passes::RemoveSingleUse0DTensors(g);
  passes::RemoveUnnecessaryCasts(g);
  passes::ReplaceAtenInt(g);
  if (lower_info.converting_to_trt_engine) {
    passes::RemoveCollectionCast(g);
  }
  passes::UnpackAndCastMaskedFill(g, lower_info.getGPUDeviceString());
  passes::UnpackAndCastNumToTensor(g, lower_info.getGPUDeviceString());
  passes::UnpackAndCastFull(g, lower_info.getGPUDeviceString());
  passes::ReplaceScalarImplicit(g);
  passes::RewriteInputsWithParams(g, params);
  passes::ReplaceAtenPad(g);
  LOG_GRAPH(*g);
}

torch::jit::Module LowerModule(const torch::jit::Module& mod, std::string method_name, const LowerInfo& lower_info) {
  std::unordered_set<std::string> forced_fallback_modules(
      lower_info.forced_fallback_modules.begin(), lower_info.forced_fallback_modules.end());
  if (forced_fallback_modules.size() > 0) {
    passes::NotateModuleForFallback(mod, "", method_name, forced_fallback_modules);
    LOG_GRAPH("After MLF notation pass: " << *mod.get_method(method_name).graph());
  }
  auto mod_ = torch::jit::freeze_module(mod);
  LOG_GRAPH("After freeze: " << *mod_.get_method(method_name).graph());
  return mod_;
}

std::pair<std::shared_ptr<torch::jit::Graph>, std::vector<torch::jit::IValue>> Lower(
    const torch::jit::Module& mod,
    std::string method_name,
    const LowerInfo& lower_info) {
  LOG_DEBUG(lower_info);
  LOG_GRAPH("Before lowering: " << *mod.get_method(method_name).graph());
  auto lowered_mod = lower_info.unfreeze_module ? mod : LowerModule(mod, method_name, lower_info);
  auto g = lowered_mod.get_method(method_name).graph();

  LOG_GRAPH("LibTorch Lowering");
  auto graph_and_ivalues = torch::jit::LowerGraph(*g, lowered_mod._ivalue());

  // Go through Torch-TensorRT Lowering to reformat graph to be conversion friendly
  // and also segment for accelerators and executors (TRT-DLA, TRT-GPU  , PYT)
  // unfreeze_module is used to not perform constant folding on weights in the network.
  // In quantization aware trained (QAT) models, weights are passed through quantize and
  // dequantize nodes which should not be folded. So unfreeze_module is set to True for QAT models.
  LOG_GRAPH("Torch-TensorRT.TorchScript Graph Lowering");
  lowering::LowerGraph(graph_and_ivalues.first, graph_and_ivalues.second, lower_info);

  // Is this necessary?
  // lowering::LowerBlock(g->block());

  LOG_INFO("Lowered Graph: " << *(graph_and_ivalues.first));
  return graph_and_ivalues;
}

} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
