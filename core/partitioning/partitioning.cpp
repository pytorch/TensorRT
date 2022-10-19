#include "core/partitioning/partitioning.h"
#include <queue>
#include "core/conversion/conversion.h"
#include "core/conversion/evaluators/evaluators.h"
#include "torch/csrc/jit/passes/constant_pooling.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"

namespace torch_tensorrt {
namespace core {
namespace partitioning {

struct usage_info {
  size_t produce_id; // id of segmented block which contains a raw value of a given torch::jit::Value
  std::vector<size_t> torch_use_id; // ids of segmented blocks which are of type Pytorch
  std::vector<size_t> tensorrt_use_id; // ids of segmented blocks which are of type TensorRT
};

inline bool isTensor(torch::jit::Value* val) {
  return val->type()->isSubtypeOf(torch::jit::TensorType::get());
}

bool containNonTensorOutputs(torch::jit::Node* n) {
  for (auto output : n->outputs()) {
    if (!isTensor(output)) {
      return true;
    }
  }
  return false;
}

// Check if the inputs and outputs of the graph are Tensor. If not, then fallback connected nodes
void setInputsOutputsConnectedNodes(PartitioningCtx* ctx, torch::jit::Block* block) {
  // fallback nodes that produce entire graph's nonTensor output
  for (auto i : block->outputs()) {
    if (!isTensor(i)) {
      ctx->setNodeExecutorDecision(i->node(), NodeExecutorDecision::kNON_TENSOR);
    }
  }

  // fallback nodes that consume entire graph's nonTensor input
  for (auto i : block->inputs()) {
    if (!isTensor(i)) {
      for (auto use : i->uses()) {
        ctx->setNodeExecutorDecision(use.user, NodeExecutorDecision::kNON_TENSOR);
      }
    }
  }
}

// Find and set all explicit fallback nodes (nodes that are unsupported or forced fallback)
// we use a map to indicate the reason why it's fallback to torch
// For any node that's not explicitly fallback, we set it to run in TensorRT for now
void setExplicitFallbackNodes(PartitioningCtx* ctx, torch::jit::Block* block) {
  auto nodes = block->nodes();
  const auto to_compile_sym = c10::Symbol::attr("to_compile");

  for (const auto n : nodes) {
    if (n->kind() == torch::jit::prim::Constant) {
      continue;
    }

    if (!conversion::OpSupported(n)) {
      // If the op is not supported by the conversion phase it should run in PyTorch
      ctx->setNodeExecutorDecision(n, NodeExecutorDecision::kUNSUPPORTED);
    } else if (ctx->forced_fallback_ops.find(n->kind().toQualString()) != ctx->forced_fallback_ops.end()) {
      // If the user specifies the op to run in Torch it should run in PyTorch
      ctx->setNodeExecutorDecision(n, NodeExecutorDecision::kOPERATOR_FALLBACK);
    } else if (n->hasAttribute(to_compile_sym) && n->i(to_compile_sym) == (int64_t) false) {
      // If the user specifies the module containing this op to run in torch it should run in PyTorch
      ctx->setNodeExecutorDecision(n, NodeExecutorDecision::kMODULE_FALLBACK);
    } else {
      // Set the rest nodes to TensorRt
      ctx->setNodeExecutorDecision(n, NodeExecutorDecision::kCONVERT);
    }
  }
  return;
}

// For a given set of fallback nodes, check their inputs/outputs, if any inputs/outputs of them are NonTensor,
// then the nodes that produces/consumes those values should also fallback
void setNonTensorConnectedNodes(PartitioningCtx* ctx, std::vector<torch::jit::Node*>& initial_fallback_nodes) {
  // initial_fallback_nodes are the fallback nodes that we have before we run BFS in this function
  std::queue<torch::jit::Node*> q;
  for (auto& node : initial_fallback_nodes) {
    q.push(node);
  }

  while (!q.empty()) {
    auto cur_node = q.front();
    q.pop();
    // for every node that produces this fallback node's NonTensor input, they should fallback too
    for (auto input : cur_node->inputs()) {
      if (!isTensor(input) && input->node()->kind() != torch::jit::prim::Constant &&
          ctx->shouldNodeRunInTensorRT(input->node())) {
        ctx->setNodeExecutorDecision(input->node(), NodeExecutorDecision::kNON_TENSOR);
        q.push(input->node());
      }
    }
    // for every node that consumes this fallback node's NonTensor output, they should fallback too
    for (auto output : cur_node->outputs()) {
      if (!isTensor(output)) {
        for (auto use : output->uses()) {
          auto node = use.user;
          if (node->kind() != torch::jit::prim::Constant && ctx->shouldNodeRunInTensorRT(node)) {
            ctx->setNodeExecutorDecision(node, NodeExecutorDecision::kNON_TENSOR);
            q.push(node);
          }
        }
      }
    }
  }
}

// Sub-function that traverses the entire block and check if TensorRT node sequence satisfy min_block_size
std::vector<torch::jit::Node*> traverseNodesForMinBlockSize(PartitioningCtx* ctx, torch::jit::Block* block) {
  auto nodes = block->nodes();
  std::vector<torch::jit::Node*> cur_trt_nodes;
  std::vector<torch::jit::Node*> min_block_fallback_nodes;
  for (const auto n : nodes) {
    if (n->kind() == torch::jit::prim::Constant) {
      continue;
    }

    // check if current node fallback or not
    if (!ctx->shouldNodeRunInTorch(n)) {
      cur_trt_nodes.push_back(n);
    } else {
      if (cur_trt_nodes.size() < ctx->settings.min_block_size) {
        min_block_fallback_nodes.insert(min_block_fallback_nodes.end(), cur_trt_nodes.begin(), cur_trt_nodes.end());
      }
      cur_trt_nodes.clear();
    }
  }
  if (cur_trt_nodes.size() < ctx->settings.min_block_size) {
    min_block_fallback_nodes.insert(min_block_fallback_nodes.end(), cur_trt_nodes.begin(), cur_trt_nodes.end());
  }
  return min_block_fallback_nodes;
}

// Set the nodes that fallback because of min_block_size
void setMinBlockFallbackNodes(PartitioningCtx* ctx, torch::jit::Block* block) {
  // first traverse all the nodes to find the initial nodes that don't meet the min_block_size requirement
  auto min_block_fallback_nodes = traverseNodesForMinBlockSize(ctx, block);

  // keep fallback until all segments meet the min_block_size requirement
  while (!min_block_fallback_nodes.empty()) {
    for (const auto i : min_block_fallback_nodes) {
      ctx->setNodeExecutorDecision(i, NodeExecutorDecision::kMIN_BLOCK_FALLBACK);
    }
    // find the fallback nodes because of dependency with min_block_size caused fallback nodes
    setNonTensorConnectedNodes(ctx, min_block_fallback_nodes);
    // keep traverse the graph until there is no node fallback because of min_block_size
    min_block_fallback_nodes = traverseNodesForMinBlockSize(ctx, block);
  }
}

bool isModifyingNodes(torch::jit::Node* node, torch::jit::Value* val) {
  const torch::jit::FunctionSchema* schema = node->maybeSchema();
  if (!schema) {
    return false;
  }
  for (size_t i = 0; i < node->inputs().size(); ++i) {
    if (node->inputs()[i] == val) {
      const at::AliasInfo* formal = schema->arguments()[i].alias_info();
      if (formal && formal->isWrite()) {
        LOG_GRAPH(
            util::node_info(node) << " is a modifying node for value " << val->debugName()
                                  << ", add it to the dependency graph.");
        return true;
      }
    }
  }
  return false;
}

std::vector<torch::jit::Node*> findModifyingNodes(
    torch::jit::Value* val,
    const std::unordered_set<torch::jit::Node*>& seg_block_nodes) {
  std::vector<torch::jit::Node*> modifying_nodes;
  for (auto use : val->uses()) {
    torch::jit::Node* node = use.user;
    if (seg_block_nodes.find(node) != seg_block_nodes.end()) {
      break;
    }
    if (isModifyingNodes(node, val)) {
      modifying_nodes.push_back(node);
    }
  }
  return modifying_nodes;
}

// this function is only used when a TRT segment produces nonTensor values which are used by later TRT segment
std::vector<torch::jit::Node*> getDependencyNodes(
    const std::vector<torch::jit::Value*>& vals,
    const SegmentedBlock& seg_block) {
  // get all nodes in the segmentedblock
  std::unordered_set<torch::jit::Node*> seg_block_nodes(seg_block.raw_nodes().begin(), seg_block.raw_nodes().end());
  // use bfs to get the DAG dependency nodes for input value
  std::queue<torch::jit::Value*, std::deque<torch::jit::Value*>> q(
      std::deque<torch::jit::Value*>(vals.begin(), vals.end()));
  std::unordered_set<torch::jit::Node*> visited;
  std::vector<torch::jit::Node*> stk;
  while (!q.empty()) {
    auto cur_val = q.front();
    q.pop();
    auto node = cur_val->node();
    if (node->kind() != torch::jit::prim::Constant && !visited.count(node)) {
      visited.insert(node);
      auto modifying_nodes = findModifyingNodes(cur_val, seg_block_nodes);
      stk.insert(stk.end(), modifying_nodes.rbegin(), modifying_nodes.rend());
      stk.push_back(node);
      for (auto input : node->inputs()) {
        if (!isTensor(input)) {
          q.push(input);
        }
      }
    }
  }
  std::reverse(stk.begin(), stk.end());
  return stk;
}

void resolveTRTNonTensorInputs(PartitioningCtx* ctx, torch::jit::Block* block) {
  // if a TRT segment has nonTensor Inputs, the nodes that produce this nonTensor Inputs must in another TensorRT engine
  // because we have already found the interface between Torch and TRT in segmentation phase
  // what we do here is just find the dependency nodes of the TRT segments that have nonTensor inputs
  PartitionedGraph& cur_partitioned_block = ctx->partitioned_blocks[block];
  for (size_t i = 0; i < cur_partitioned_block.size(); ++i) {
    if (cur_partitioned_block[i].target() == SegmentedBlock::kTensorRT) {
      std::vector<torch::jit::Value*> inputs_to_resolve;
      for (auto input : cur_partitioned_block[i].raw_inputs()) {
        if (!isTensor(input)) {
          inputs_to_resolve.push_back(input);
        }
      }
      if (!inputs_to_resolve.empty()) {
        std::vector<torch::jit::Node*> dependency_nodes =
            getDependencyNodes(inputs_to_resolve, cur_partitioned_block[i]);
        dependency_nodes.insert(
            dependency_nodes.end(),
            cur_partitioned_block[i].raw_nodes().begin(),
            cur_partitioned_block[i].raw_nodes().end());
        cur_partitioned_block[i] = SegmentedBlock(SegmentedBlock::kTensorRT, dependency_nodes);
      }
    }
  }
}

void registerSegmentsOutputs(PartitioningCtx* ctx, torch::jit::Block* block) {
  // find the corresponding raw values in original global graph for this segmented block's inputs/outputs
  PartitionedGraph& cur_partitioned_block = ctx->partitioned_blocks[block];
  auto cmp = [](torch::jit::Value* a, torch::jit::Value* b) { return a->unique() < b->unique(); };
  std::set<torch::jit::Value*, decltype(cmp)> input_values(cmp);
  for (auto& seg_block : cur_partitioned_block) {
    for (auto& input : seg_block.raw_inputs()) {
      input_values.insert(input);
    }
  }

  for (auto& graph_output : block->outputs()) {
    input_values.insert(graph_output);
  }

  // should be careful here because some in-place operations don't return any values, there is no output for this kind
  // of segment identify the output for each mini-graph by checking if any value in this graph is used later we
  // shouldn't register nonTensor output for TensorRT segments
  for (auto& seg_block : cur_partitioned_block) {
    for (auto& mini_graph_input : input_values) {
      if (std::find(seg_block.raw_inputs().begin(), seg_block.raw_inputs().end(), mini_graph_input) ==
              seg_block.raw_inputs().end() &&
          seg_block.contain_raw_value(mini_graph_input)) {
        if (!isTensor(mini_graph_input) && seg_block.target() == SegmentedBlock::kTensorRT)
          continue;
        seg_block.registerOutput(mini_graph_input);
      }
    }
    // if no output, then register the last node's output as current graph's output
    if (seg_block.raw_outputs().empty()) {
      // for Torch segments, register input as output
      if (seg_block.target() == SegmentedBlock::kTorch) {
        seg_block.registerOutput(seg_block.raw_inputs()[0]);
      } else {
        // for TensorRT segments, register last nonInput Tensor outputs
        for (int i = seg_block.raw_nodes().size() - 1; i >= 0; --i) {
          for (auto node_output : seg_block.raw_nodes()[i]->outputs()) {
            if (isTensor(node_output))
              seg_block.registerOutput(node_output);
          }
          if (!seg_block.raw_outputs().empty())
            break;
        }
      }
    }
  }

  std::for_each(cur_partitioned_block.begin(), cur_partitioned_block.end(), [](SegmentedBlock& seg_block) {
    torch::jit::EliminateDeadCode(seg_block.g());
  });
  // erase segments which still have no output
  cur_partitioned_block.erase(
      std::remove_if(
          cur_partitioned_block.begin(),
          cur_partitioned_block.end(),
          [](SegmentedBlock& seg_block) { return seg_block.raw_outputs().empty(); }),
      cur_partitioned_block.end());

  return;
}

// Need to check if this makes sense might be a root cause of some issues of over aggressive fallback
bool checkLoopEvaluatable(torch::jit::Node* n) {
  bool compile_to_trt = true;
  for (auto bn : n->blocks()[0]->nodes()) {
    if (bn->kind() == torch::jit::prim::Loop) {
      compile_to_trt = compile_to_trt && checkLoopEvaluatable(bn);
    } else if (bn->kind() == torch::jit::prim::If) {
      compile_to_trt = compile_to_trt && containNonTensorOutputs(bn);
    } else {
      compile_to_trt = compile_to_trt && core::conversion::evaluators::shouldEvalAtConversionTime(bn);
    }
  }
  return compile_to_trt;
}

void finalizeNewBlock(
    PartitionedGraph& g,
    SegmentedBlock::SegmentedBlockTarget kind,
    std::vector<torch::jit::Node*>& nodes) {
  LOG_DEBUG("Finalizing in progress " << SegmentedBlock::target_to_str(kind) << " block");
  g.emplace_back(g.size(), kind, nodes);
  nodes.clear();
  LOG_DEBUG(g.back());
}

void setNodeExecutorLUT(PartitioningCtx* ctx, torch::jit::Block* block) {
  // First, find all the explicit fallback nodes that should run in Torch:
  // 1. nodes that are unsupported
  // 2. nodes that the user specifies to run in torch
  // 3. nodes that the user specifies the module containing this op to run in torch
  // At the same time, set all the rest nodes to NodeExecutorDecision::kCONVERT
  setExplicitFallbackNodes(ctx, block);

  // Second, check if there is nonTensor input/output for the block, if there is, then fallback the nodes that
  // consume/produce this nonTensor value
  setInputsOutputsConnectedNodes(ctx, block);

  // Third, for fallback nodes, if it consumes any NonTensor inputs, then the nodes that produce this
  // input should also fallback. Similarly, if it produces any NonTensor outputs, then the nodes
  // that consume this output should also fallback
  auto cur_fallback_nodes = ctx->getNodesRunInTorch();
  setNonTensorConnectedNodes(ctx, cur_fallback_nodes);

  // Finally, check if all current tensorrt blocks satisfy the min_block_size requirement.
  // We need to traverse the whole graph many times here
  setMinBlockFallbackNodes(ctx, block);
}

void segmentGraph(PartitioningCtx* ctx, torch::jit::Block* block) {
  // Find all the fallback nodes and build execution decision LUT for all nodes
  setNodeExecutorLUT(ctx, block);

  auto nodes = block->nodes();

  // segment the nodes
  PartitionedGraph segmented_blocks;

  std::vector<torch::jit::Node*> in_prog_trt_blk_nodes, in_prog_pyt_blk_nodes;
  for (const auto n : nodes) {
    // Skip constant nodes as they are resources for both kinds of modules
    if (n->kind() == torch::jit::prim::Constant) {
      continue;
    }
    // the outputs of trt subgraph shouldn't be collections
    if (ctx->shouldNodeRunInTensorRT(n)) {
      in_prog_trt_blk_nodes.push_back(n);

      // If there is an active PyTorch block and we have passed the threshold for a valid TRT
      // block then segment and reset the active PyTorch block
      if (in_prog_trt_blk_nodes.size() >= ctx->settings.min_block_size && !in_prog_pyt_blk_nodes.empty()) {
        finalizeNewBlock(segmented_blocks, SegmentedBlock::kTorch, in_prog_pyt_blk_nodes);
      }
    } else {
      // If there is an active TRT block that is valid segment and reset the active TRT block
      // otherwise add it to the active PyTorch block and reset
      if (in_prog_trt_blk_nodes.size() >= ctx->settings.min_block_size) {
        finalizeNewBlock(segmented_blocks, SegmentedBlock::kTensorRT, in_prog_trt_blk_nodes);
      } else {
        LOG_DEBUG(
            "In progress TRT block does not meet minimum block size requirements ("
            << in_prog_trt_blk_nodes.size() << ", expected at least " << ctx->settings.min_block_size
            << "), therefore folding into in progress PyTorch block");
        in_prog_pyt_blk_nodes.insert(
            in_prog_pyt_blk_nodes.end(), in_prog_trt_blk_nodes.begin(), in_prog_trt_blk_nodes.end());
      }
      in_prog_trt_blk_nodes.clear();
      // if there is a prim::If then this if node will be encapsulated in a SegmentedBlock
      // we shouldn't inject node for this block in dependency analysis process
      if (n->kind() == torch::jit::prim::If) {
        LOG_DEBUG(
            "Hit a conditional statement, finializing in progress PYT block and creating a new one for the conditional");
        if (!in_prog_pyt_blk_nodes.empty()) {
          finalizeNewBlock(segmented_blocks, SegmentedBlock::kTorch, in_prog_pyt_blk_nodes);
        }
        auto cond_node = std::vector<torch::jit::Node*>{n};
        finalizeNewBlock(segmented_blocks, SegmentedBlock::kTorch, cond_node);
        continue;
      } else if (n->kind() == torch::jit::prim::Loop) {
        if (!in_prog_pyt_blk_nodes.empty()) {
          finalizeNewBlock(segmented_blocks, SegmentedBlock::kTorch, in_prog_pyt_blk_nodes);
        }
        if (checkLoopEvaluatable(n)) {
          in_prog_trt_blk_nodes.push_back(n);
        } else {
          auto loop_node = std::vector<torch::jit::Node*>{n};
          finalizeNewBlock(segmented_blocks, SegmentedBlock::kTorch, loop_node);
        }
        continue;
      }
      in_prog_pyt_blk_nodes.push_back(n);
    }
  }

  // if there is any kTorch nodes left, then either the last nodes are kTorch or last nodes are kTensorRT but num <
  // min_block_size
  if (in_prog_trt_blk_nodes.size() >= ctx->settings.min_block_size) {
    finalizeNewBlock(segmented_blocks, SegmentedBlock::kTensorRT, in_prog_trt_blk_nodes);
  }

  if (!in_prog_pyt_blk_nodes.empty() || !in_prog_trt_blk_nodes.empty()) {
    in_prog_pyt_blk_nodes.insert(
        in_prog_pyt_blk_nodes.end(), in_prog_trt_blk_nodes.begin(), in_prog_trt_blk_nodes.end());
    finalizeNewBlock(segmented_blocks, SegmentedBlock::kTorch, in_prog_pyt_blk_nodes);
  }

  ctx->partitioned_blocks.insert({block, segmented_blocks});
  return;
}

bool isInputDynamic(PartitioningCtx* ctx) {
  // Check if inputs have dynamic shapes
  bool input_is_dynamic = true;
  auto inputs_map = ctx->settings.collection_input_spec_map;
  for (auto inputs : inputs_map) {
    for (auto input : inputs.second) {
      if (!input.input_is_dynamic) {
        input_is_dynamic = false;
      }
    }
  }
  return input_is_dynamic;
}

void partition(PartitioningCtx* ctx) {
  LOG_DEBUG(ctx->settings);

  // Go through all the blocks to do the partitioning
  for (torch::jit::Block* block : ctx->original_blocks) {
    // segment lowering global graph into blocks
    segmentGraph(ctx, block);

    // It's possible that some TensorRT blocks have nonTensor inputs/output because they are interleaved by Torch blocks
    // resolve nonTensor inputs/outputs
    LOG_DEBUG("Resolving non-tensor inputs for segmented blocks");
    resolveTRTNonTensorInputs(ctx, block);

    // register input/output torch::jit::Value for segmented graphs
    LOG_DEBUG("Registering input/output torch::jit::Value for segmented graphs");
    registerSegmentsOutputs(ctx, block);

    // Incase of dynamic shape inputs, run shape analysis on each segmented block for min/opt/max ranges and register
    // output shapes for each block accordingly
    if (isInputDynamic(ctx)) {
      LOG_DEBUG("Performing shape analysis for segmented blocks using min/opt/max shapes for inputs");
      auto min_input_ivalues_map =
          partitioning::generateRandomInputs(ctx->settings.collection_input_spec_map, ctx->input_types_map, "min");
      auto opt_input_ivalues_map =
          partitioning::generateRandomInputs(ctx->settings.collection_input_spec_map, ctx->input_types_map, "opt");
      auto max_input_ivalues_map =
          partitioning::generateRandomInputs(ctx->settings.collection_input_spec_map, ctx->input_types_map, "max");

      runShapeAnalysis(ctx, block, min_input_ivalues_map, "min");
      runShapeAnalysis(ctx, block, opt_input_ivalues_map, "opt");
      runShapeAnalysis(ctx, block, max_input_ivalues_map, "max");
    } else {
      LOG_DEBUG("Performing shape analysis for segmented blocks using static shapes for inputs");
      auto opt_input_ivalues_map =
          partitioning::generateRandomInputs(ctx->settings.collection_input_spec_map, ctx->input_types_map, "opt");
      runShapeAnalysis(ctx, block, opt_input_ivalues_map, "opt");
    }
  }
}

} // namespace partitioning
} // namespace core
} // namespace torch_tensorrt
