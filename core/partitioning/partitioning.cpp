#include "partitioning.h"

#include <queue>
#include "core/conversion/conversion.h"
#include "core/conversion/evaluators/evaluators.h"
#include "core/partitioning/shape_analysis.h"
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

// check if the input and output of the graph is Tensor after collection is enabled. If it is, then fallback related
// nodes
void fallback_graph_nontensor_in_out(
    torch::jit::Block* block,
    std::unordered_map<torch::jit::Node*, int>& global_fallback_nodes) {
  // fallback nodes that produce entire graph's nonTensor output
  for (auto i : block->outputs()) {
    if (!isTensor(i)) {
      global_fallback_nodes.insert({i->node(), FallbackNodeType::kNON_TENSOR});
    }
  }

  // fallback nodes that consume entire graph's nonTensor input
  for (auto i : block->inputs()) {
    if (!isTensor(i)) {
      for (auto use : i->uses()) {
        global_fallback_nodes.insert({use.user, FallbackNodeType::kNON_TENSOR});
      }
    }
  }
}

void find_all_fallback_nodes(
    std::unordered_map<torch::jit::Node*, int>& initial_fallback_nodes,
    std::unordered_map<torch::jit::Node*, int>& global_fallback_nodes) {
  // initial_fallback_nodes are the fallback nodes that we have before we run BFS in this function
  // global_fallback_nodes are the fallback nodes that we maintain globally
  std::queue<torch::jit::Node*> q;
  for (auto& node : initial_fallback_nodes) {
    q.push(node.first);
  }

  std::unordered_set<torch::jit::Node*> visited_nodes;
  while (!q.empty()) {
    auto cur_node = q.front();
    q.pop();
    // for every node that produces this fallback node's NonTensor input, they should fallback too
    for (auto input : cur_node->inputs()) {
      if (!isTensor(input) && input->node()->kind() != torch::jit::prim::Constant &&
          global_fallback_nodes.insert({input->node(), FallbackNodeType::kNON_TENSOR}).second) {
        q.push(input->node());
      }
    }
    // for every node that consumes this fallback node's NonTensor output, they should fallback too
    for (auto output : cur_node->outputs()) {
      if (!isTensor(output)) {
        for (auto use : output->uses()) {
          auto node = use.user;
          if (node->kind() != torch::jit::prim::Constant &&
              global_fallback_nodes.insert({node, FallbackNodeType::kNON_TENSOR}).second) {
            q.push(node);
          }
        }
      }
    }
  }
}

void resolveTRTNonTensorInputs(PartitionedGraph& segmented_blocks) {
  // if a TRT segment has nonTensor Inputs, the nodes that produce this nonTensor Inputs must in another TensorRT engine
  // because we have already found the interface between Torch and TRT in segmentation phase
  // what we do here is just find the dependency nodes of the TRT segments that have nonTensor inputs
  for (size_t i = 0; i < segmented_blocks.size(); ++i) {
    if (segmented_blocks[i].target() == SegmentedBlock::kTensorRT) {
      std::vector<torch::jit::Value*> inputs_to_resolve;
      for (auto input : segmented_blocks[i].raw_inputs()) {
        if (!isTensor(input)) {
          inputs_to_resolve.push_back(input);
        }
      }
      if (!inputs_to_resolve.empty()) {
        std::vector<torch::jit::Node*> dependency_nodes = getDependencyNodes(inputs_to_resolve, segmented_blocks[i]);
        dependency_nodes.insert(
            dependency_nodes.end(), segmented_blocks[i].raw_nodes().begin(), segmented_blocks[i].raw_nodes().end());
        segmented_blocks[i] = SegmentedBlock(SegmentedBlock::kTensorRT, dependency_nodes);
      }
    }
  }
}

void registerSegmentsOutputs(PartitionedGraph& segmented_blocks, torch::jit::Block* block) {
  // find the corresponding raw values in original global graph for this segmented block's inputs/outputs
  auto cmp = [](torch::jit::Value* a, torch::jit::Value* b) { return a->unique() < b->unique(); };
  std::set<torch::jit::Value*, decltype(cmp)> input_values(cmp);
  for (auto& seg_block : segmented_blocks) {
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
  for (auto& seg_block : segmented_blocks) {
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

  std::for_each(segmented_blocks.begin(), segmented_blocks.end(), [](SegmentedBlock& seg_block) {
    torch::jit::EliminateDeadCode(seg_block.g());
  });
  // erase segments which still have no output
  segmented_blocks.erase(
      std::remove_if(
          segmented_blocks.begin(),
          segmented_blocks.end(),
          [](SegmentedBlock& seg_block) { return seg_block.raw_outputs().empty(); }),
      segmented_blocks.end());

  return;
}

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

bool check_node_fallback(torch::jit::Node* n, const std::unordered_map<torch::jit::Node*, int>& fallback_nodes) {
  if (fallback_nodes.count(n)) {
    if (fallback_nodes.at(n) == FallbackNodeType::kUNSUPPORTED) {
      LOG_GRAPH("Node not supported by conversion: " << util::node_info(n));
    } else if (fallback_nodes.at(n) == FallbackNodeType::kOPERATOR_FALLBACK) {
      LOG_GRAPH("Node explicitly set to run in torch: " << util::node_info(n));
    } else if (fallback_nodes.at(n) == FallbackNodeType::kMODULE_FALLBACK) {
      LOG_GRAPH("Node is within a module set to run in torch: " << util::node_info(n));
    } else if (fallback_nodes.at(n) == FallbackNodeType::kMIN_BLOCK_FALLBACK) {
      LOG_GRAPH("Node fallback to Torch because of min_block_size" << util::node_info(n));
    } else {
      LOG_GRAPH(
          "Node fallback to Torch because the NonTensor dependencies with other fallback nodes: "
          << util::node_info(n));
    }
    return false;
  }

  LOG_GRAPH("Node is going to run in TensorRT: " << util::node_info(n));
  return true;
}

void finalize_block(
    PartitionedGraph& g,
    SegmentedBlock::SegmentedBlockTarget kind,
    std::vector<torch::jit::Node*>& nodes) {
  LOG_DEBUG("Finalizing in progress " << SegmentedBlock::target_to_str(kind) << " block");
  g.emplace_back(g.size(), kind, nodes);
  nodes.clear();
  LOG_DEBUG(g.back());
}

// use this function to get all initial fallback nodes (nodes that are unsupported or forced fallback)
// we use a map to indicate the reason why it's fallback to torch
void get_fallback_nodes(
    torch::jit::Block* block,
    const std::unordered_set<std::string>& forced_fallback_ops,
    std::unordered_map<torch::jit::Node*, int>& fallback_nodes) {
  auto nodes = block->nodes();
  for (const auto n : nodes) {
    if (n->kind() == torch::jit::prim::Constant) {
      continue;
    }

    // If the op is not supported by the conversion phase it should run in PyTorch
    if (!conversion::OpSupported(n)) {
      fallback_nodes.insert({n, FallbackNodeType::kUNSUPPORTED});
    }

    // If the user specifies the op to run in Torch it should run in PyTorch
    if (forced_fallback_ops.find(n->kind().toQualString()) != forced_fallback_ops.end()) {
      fallback_nodes.insert({n, FallbackNodeType::kOPERATOR_FALLBACK});
    }

    // If the user specifies the module containing this op to run in torch it should run in PyTorch
    const auto to_compile_sym = c10::Symbol::attr("to_compile");
    if (n->hasAttribute(to_compile_sym) && n->i(to_compile_sym) == (int64_t) false) {
      fallback_nodes.insert({n, FallbackNodeType::kMODULE_FALLBACK});
    }
  }
  return;
}

std::vector<torch::jit::Node*> traverse_nodes_for_min_block_size(
    torch::jit::Block* block,
    const std::unordered_map<torch::jit::Node*, int>& global_fallback_nodes,
    size_t min_block_size) {
  auto nodes = block->nodes();
  std::vector<torch::jit::Node*> cur_trt_nodes;
  std::vector<torch::jit::Node*> min_block_fallback_nodes;
  for (const auto n : nodes) {
    if (n->kind() == torch::jit::prim::Constant)
      continue;

    // check if current node fallback or not
    if (!global_fallback_nodes.count(n)) {
      // if this node is not in fallback nodes, then it's in trt segments
      cur_trt_nodes.push_back(n);
    } else {
      if (cur_trt_nodes.size() < min_block_size) {
        min_block_fallback_nodes.insert(min_block_fallback_nodes.end(), cur_trt_nodes.begin(), cur_trt_nodes.end());
      }
      cur_trt_nodes.clear();
    }
  }
  if (cur_trt_nodes.size() < min_block_size) {
    min_block_fallback_nodes.insert(min_block_fallback_nodes.end(), cur_trt_nodes.begin(), cur_trt_nodes.end());
  }
  return min_block_fallback_nodes;
}

void find_min_block_size_fallback_nodes(
    torch::jit::Block* block,
    std::unordered_map<torch::jit::Node*, int>& global_fallback_nodes,
    size_t min_block_size) {
  // first traverse all the nodes to find the initial nodes that don't meet the min_block_size requirement
  auto min_block_fallback_nodes = traverse_nodes_for_min_block_size(block, global_fallback_nodes, min_block_size);
  std::unordered_map<torch::jit::Node*, int> initial_fallback_nodes;

  // keep fallback until all segments meet the min_block_size requirement
  while (!min_block_fallback_nodes.empty()) {
    for (const auto i : min_block_fallback_nodes) {
      initial_fallback_nodes.insert({i, FallbackNodeType::kMIN_BLOCK_FALLBACK});
    }
    global_fallback_nodes.insert(initial_fallback_nodes.begin(), initial_fallback_nodes.end());
    // find the fallback nodes because of dependency with min_block_size caused fallback nodes
    find_all_fallback_nodes(initial_fallback_nodes, global_fallback_nodes);
    // keep traverse the graph until there is no node fallback because of min_block_size
    min_block_fallback_nodes = traverse_nodes_for_min_block_size(block, global_fallback_nodes, min_block_size);
  }
}

PartitionedGraph segment_graph(
    torch::jit::Block* block,
    const PartitionInfo& partition_info,
    std::unordered_map<torch::jit::Node*, int>& global_fallback_nodes) {
  auto min_block_size = partition_info.min_block_size;
  std::unordered_set<std::string> forced_fallback_ops(
      partition_info.forced_fallback_operators.begin(), partition_info.forced_fallback_operators.end());

  // get the initial fallback nodes (nodes that are unsupported or forced fallback)
  get_fallback_nodes(block, forced_fallback_ops, global_fallback_nodes);

  // For fallback nodes, if it consumes any NonTensor inputs or TensorList inputs, then the node that produces this
  // input should also fallback Similarly, if it produces any NonTensor outputs or TensorList outputs, then the node
  // that produces this input should also fallback
  // TODO: don't need to fallback the TensorList related nodes once the collection feature is supported
  find_all_fallback_nodes(global_fallback_nodes, global_fallback_nodes);

  // find all fallback nodes because of the min_block_size requirement
  find_min_block_size_fallback_nodes(block, global_fallback_nodes, min_block_size);

  auto nodes = block->nodes();
  PartitionedGraph segmented_blocks;

  // segment the nodes
  std::vector<torch::jit::Node*> in_prog_trt_blk_nodes, in_prog_pyt_blk_nodes;
  for (const auto n : nodes) {
    // Skip constant nodes as they are resources for both kinds of modules
    if (n->kind() == torch::jit::prim::Constant) {
      continue;
    }
    // the outputs of trt subgraph shouldn't be collections
    if (check_node_fallback(n, global_fallback_nodes)) {
      in_prog_trt_blk_nodes.push_back(n);

      // If there is an active PyTorch block and we have passed the threshold for a valid TRT
      // block then segment and reset the active PyTorch block
      if (in_prog_trt_blk_nodes.size() >= min_block_size && !in_prog_pyt_blk_nodes.empty()) {
        finalize_block(segmented_blocks, SegmentedBlock::kTorch, in_prog_pyt_blk_nodes);
      }
    } else {
      // If there is an active TRT block that is valid segment and reset the active TRT block
      // otherwise add it to the active PyTorch block and reset
      if (in_prog_trt_blk_nodes.size() >= min_block_size) {
        finalize_block(segmented_blocks, SegmentedBlock::kTensorRT, in_prog_trt_blk_nodes);
      } else {
        LOG_DEBUG(
            "In progress TRT block does not meet minimum block size requirements, therefore folding into in progress PyTorch block");
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
          finalize_block(segmented_blocks, SegmentedBlock::kTorch, in_prog_pyt_blk_nodes);
        }
        auto cond_node = std::vector<torch::jit::Node*>{n};
        finalize_block(segmented_blocks, SegmentedBlock::kTorch, cond_node);
        continue;
      } else if (n->kind() == torch::jit::prim::Loop) {
        if (!in_prog_pyt_blk_nodes.empty()) {
          finalize_block(segmented_blocks, SegmentedBlock::kTorch, in_prog_pyt_blk_nodes);
        }
        if (checkLoopEvaluatable(n)) {
          in_prog_trt_blk_nodes.push_back(n);
        } else {
          auto loop_node = std::vector<torch::jit::Node*>{n};
          finalize_block(segmented_blocks, SegmentedBlock::kTorch, loop_node);
        }
        continue;
      }
      in_prog_pyt_blk_nodes.push_back(n);
    }
  }

  // if there is any kTorch nodes left, then either the last nodes are kTorch or last nodes are kTensorRT but num <
  // min_block_size
  if (in_prog_trt_blk_nodes.size() >= min_block_size) {
    finalize_block(segmented_blocks, SegmentedBlock::kTensorRT, in_prog_trt_blk_nodes);
  }

  if (!in_prog_pyt_blk_nodes.empty() || !in_prog_trt_blk_nodes.empty()) {
    in_prog_pyt_blk_nodes.insert(
        in_prog_pyt_blk_nodes.end(), in_prog_trt_blk_nodes.begin(), in_prog_trt_blk_nodes.end());
    finalize_block(segmented_blocks, SegmentedBlock::kTorch, in_prog_pyt_blk_nodes);
  }
  return segmented_blocks;
}

PartitionedGraph Partition(
    torch::jit::Block* block,
    std::unordered_map<const torch::jit::Value*, torch::jit::IValue>& example_tensor_map,
    const PartitionInfo& partition_info,
    std::unordered_map<torch::jit::Node*, int>& global_fallback_nodes) {
  LOG_DEBUG(partition_info);
  // if there is nonTensor input/output for the entire graph, fallback the node that consumes/produces this nonTensor
  // output
  fallback_graph_nontensor_in_out(block, global_fallback_nodes);

  // segment lowering global graph into blocks
  LOG_DEBUG("Parititioning source module into PyTorch and TensorRT sub blocks");
  PartitionedGraph segmented_blocks = segment_graph(block, partition_info, global_fallback_nodes);

  // It's possible that some TensorRT blocks have nonTensor inputs/output because they are interleaved by Torch blocks

  // resolve nonTensor inputs/outputs
  resolveTRTNonTensorInputs(segmented_blocks);

  // register input/output torch::jit::Value for segmented graphs
  LOG_DEBUG("Registering input/output torch::jit::Value for segmented graphs");
  registerSegmentsOutputs(segmented_blocks, block);

  // run shape analysis on each segmented block
  runShapeAnalysis(segmented_blocks, example_tensor_map, partition_info);

  for (uint64_t i = 0; i < segmented_blocks.size(); i++) {
    segmented_blocks[i].update_id(i);
  }

  LOG_INFO(segmented_blocks);

  return segmented_blocks;
}

std::ostream& operator<<(std::ostream& os, const PartitionedGraph& g) {
  os << "Partitioned Graph: [";
  for (auto b : g) {
    os << b;
  }
  os << "]";
  return os;
}

} // namespace partitioning
} // namespace core
} // namespace torch_tensorrt
