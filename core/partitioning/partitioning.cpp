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

inline bool isTensorOrTensorList(torch::jit::Value* val) {
  return val->type()->isSubtypeOf(torch::jit::TensorType::get()) ||
      val->type()->isSubtypeOf(torch::jit::ListType::ofTensors());
}

inline bool isTensorList(torch::jit::Value* val) {
  return val->type()->isSubtypeOf(torch::jit::ListType::ofTensors());
}

inline bool isTensor(torch::jit::Value* val) {
  return val->type()->isSubtypeOf(torch::jit::TensorType::get());
}

bool isAllNodesSupported(const std::vector<torch::jit::Node*>& nodes) {
  for (auto node : nodes) {
    if (!conversion::OpSupported(node)) {
      return false;
    }
  }
  return true;
}

bool containTargetInputs(torch::jit::Node* n, const std::unordered_set<torch::jit::Value*>& target_inputs) {
  for (auto input : n->inputs()) {
    if (!isTensorOrTensorList(input) && target_inputs.count(input)) {
      return true;
    }
  }
  return false;
}

bool containNonTensorOutputs(torch::jit::Node* n) {
  for (auto output : n->outputs()) {
    if (!isTensorOrTensorList(output)) {
      return true;
    }
  }
  return false;
}

std::vector<torch::jit::Node*> getDependencyNodes(const std::vector<torch::jit::Value*>& vals) {
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
      stk.push_back(node);
      for (auto input : node->inputs()) {
        if (!isTensorOrTensorList(input)) {
          q.push(input);
        }
      }
    }
  }
  std::reverse(stk.begin(), stk.end());
  return stk;
}

std::vector<torch::jit::Node*> getOutputNodes(
    torch::jit::Value* value,
    const std::unordered_set<torch::jit::Node*>& seg_block_nodes) {
  // use bfs to get the DAG outputs nodes for input value
  std::queue<torch::jit::Value*> q;
  std::vector<torch::jit::Node*> stk;
  std::unordered_set<torch::jit::Node*> visited;
  q.push(value);

  // top-down order traversing
  while (!q.empty()) {
    auto cur_val = q.front();
    q.pop();
    for (auto use : cur_val->uses()) {
      auto node = use.user;
      // use node must be in seg_block_nodes
      if (seg_block_nodes.count(node) && !visited.count(node)) {
        stk.push_back(node);
        visited.insert(node);
        // travel its' all outputs
        for (auto output : node->outputs()) {
          if (!isTensor(output)) {
            q.push(output);
          }
        }
      }
    }
  }

  // top-down order and we don't need to reverse it
  return stk;
}

void getDirtyNodes(
    std::unordered_set<torch::jit::Node*>& dirty_nodes,
    const std::unordered_set<torch::jit::Node*>& seg_block_nodes) {
  std::queue<torch::jit::Node*> q;
  for (auto& node : dirty_nodes) {
    q.push(node);
  }
  dirty_nodes.clear();

  while (!q.empty()) {
    auto cur_node = q.front();
    q.pop();
    if (!dirty_nodes.count(cur_node) && seg_block_nodes.count(cur_node)) {
      dirty_nodes.insert(cur_node);
      for (auto input : cur_node->inputs()) {
        if (!isTensorOrTensorList(input)) {
          q.push(input->node());
        }
      }
      for (auto output : cur_node->outputs()) {
        if (!isTensorOrTensorList(output)) {
          for (auto use : output->uses()) {
            auto node = use.user;
            q.push(node);
          }
        }
      }
    }
  }
}

std::pair<std::unordered_map<torch::jit::Value*, SegmentedBlock>, SegmentedBlock> segmentBlocksWithTensorListInputs(
    SegmentedBlock& seg_block,
    const std::unordered_map<torch::jit::Value*, SegmentedBlock>& tensorlist_inputs) {
  std::unordered_set<torch::jit::Node*> all_append_nodes;
  std::unordered_map<torch::jit::Value*, SegmentedBlock> append_blocks;
  const std::unordered_set<torch::jit::Node*> seg_block_nodes(
      seg_block.raw_nodes().begin(), seg_block.raw_nodes().end());
  for (auto input_pair : tensorlist_inputs) {
    auto append_nodes = getOutputNodes(input_pair.first, seg_block_nodes);
    append_blocks[input_pair.first] = SegmentedBlock(input_pair.second.target(), append_nodes);
    all_append_nodes.insert(append_nodes.begin(), append_nodes.end());
  }

  std::vector<torch::jit::Node*> trt_nodes;
  for (auto node : seg_block.raw_nodes()) {
    if (all_append_nodes.count(node) == 0) {
      trt_nodes.emplace_back(node);
    }
  }
  SegmentedBlock trt_block(SegmentedBlock::kTensorRT, trt_nodes);

  return std::pair<std::unordered_map<torch::jit::Value*, SegmentedBlock>, SegmentedBlock>(append_blocks, trt_block);
}

PartitionedGraph segmentBlocksWithSpecifiedInputs(
    SegmentedBlock& seg_block,
    std::vector<torch::jit::Value*>& inputs_to_resolve) {
  std::vector<torch::jit::Node*> dependency_nodes = getDependencyNodes(inputs_to_resolve);
  PartitionedGraph new_seg_blocks;
  // if current block is kTorch or current block is TensorRT and all dependent nodes are also supported, merge the
  // dependency nodes at the beginning of the current segmented_block and return this merged segmented_block
  if (seg_block.target() == SegmentedBlock::kTorch || isAllNodesSupported(dependency_nodes)) {
    // if current node is prim::If, just ensure that we have all required input in kTorch
    if (seg_block.raw_nodes()[0]->kind() == torch::jit::prim::If) {
      new_seg_blocks.emplace_back(seg_block.target(), dependency_nodes);
      new_seg_blocks.push_back(seg_block);
    } else {
      dependency_nodes.insert(dependency_nodes.end(), seg_block.raw_nodes().begin(), seg_block.raw_nodes().end());
      new_seg_blocks.emplace_back(seg_block.target(), dependency_nodes);
    }
  } else {
    // if current block is kTensorRT but the dependency nodes contain unsupported node, then we have to segment again
    std::unordered_set<torch::jit::Value*> inputs_to_resolve_set(inputs_to_resolve.begin(), inputs_to_resolve.end());
    std::vector<torch::jit::Node*> tensorrt_nodes, pytorch_nodes;

    // take all nodes with non_tensor_inputs as initial dirty nodes (nodes that should be in PyTorch block), then we use
    // dfs/bfs to find all dirty nodes that consume non_tensor values produced by dirty nodes or  produces non_tensor
    // values consumed by dirty nodes
    std::unordered_set<torch::jit::Node*> dirty_nodes;
    const std::unordered_set<torch::jit::Node*> seg_block_nodes(
        seg_block.raw_nodes().begin(), seg_block.raw_nodes().end());

    for (auto n : seg_block.raw_nodes()) {
      if (containTargetInputs(n, inputs_to_resolve_set)) {
        dirty_nodes.insert(n);
      }
    }
    getDirtyNodes(dirty_nodes, seg_block_nodes);
    for (auto n : seg_block.raw_nodes()) {
      if (dirty_nodes.count(n)) {
        if (!tensorrt_nodes.empty()) {
          new_seg_blocks.emplace_back(new_seg_blocks.size(), SegmentedBlock::kTensorRT, tensorrt_nodes);
          tensorrt_nodes.clear();
        }
        pytorch_nodes.push_back(n);
      } else {
        if (!pytorch_nodes.empty()) {
          new_seg_blocks.emplace_back(new_seg_blocks.size(), SegmentedBlock::kTorch, pytorch_nodes);
          pytorch_nodes.clear();
        }
        tensorrt_nodes.push_back(n);
      }
    }

    // Form the last segmented_block with the leftover nodes in tensorrt_nodes or pytorch_nodes correspondingly.
    if (!tensorrt_nodes.empty()) {
      new_seg_blocks.emplace_back(new_seg_blocks.size(), SegmentedBlock::kTensorRT, tensorrt_nodes);
    } else {
      new_seg_blocks.emplace_back(new_seg_blocks.size(), SegmentedBlock::kTorch, pytorch_nodes);
    }
  }

  return new_seg_blocks;
}

PartitionedGraph segmentBlocksWithNonTensorInputs(SegmentedBlock& seg_block) {
  // reconstruct segmented_block if this block requires nonTensor input
  std::vector<torch::jit::Value*> inputs_to_resolve;
  // Gather all non-tensor inputs for this block
  for (auto input : seg_block.raw_inputs()) {
    if (!isTensorOrTensorList(input)) {
      inputs_to_resolve.push_back(input);
    }
  }
  return segmentBlocksWithSpecifiedInputs(seg_block, inputs_to_resolve);
}

std::unordered_map<torch::jit::Value*, usage_info> getInputUsageCounts(
    const PartitionedGraph& segmented_blocks,
    const std::function<bool(torch::jit::Value*)>& condition) {
  // usage_counts is a map which stores non-tensor inputs as keys and the values are indices of segmented blocks which
  // have these non-tensor inputs. Iterate through the graph (segmented blocks) from bottom to top. When we find a
  // non-tensor input in a segmented block of index "i", store it in the usage_counts map. Now for each non-tensor
  // inputs recorded in the usage_counts map, we check if any previous segmented block (segmented block index i goes
  // from n-1 to 0) generated/contains this non-tensor input. If so, we set this idx as the produce_id as it produces
  // the non-tensor input.
  std::unordered_map<torch::jit::Value*, usage_info> usage_counts;
  for (int i = segmented_blocks.size() - 1; i >= 0; --i) {
    for (auto input : segmented_blocks[i].raw_inputs()) {
      if (condition(input)) {
        segmented_blocks[i].target() == SegmentedBlock::kTorch ? usage_counts[input].torch_use_id.push_back(i)
                                                               : usage_counts[input].tensorrt_use_id.push_back(i);
      }
    }

    // For each non-tensor value in the usage_counts map, keep updating the produce_id to the earliest segmented block
    // that has/produces it.
    for (auto& use : usage_counts) {
      // Set the produce_id to the segmented block index that contains/produces this non-tensor torch::jit::Value
      if (segmented_blocks[i].contain_raw_value(use.first)) {
        use.second.produce_id = i;
      }
    }
  }
  return usage_counts;
}

std::unordered_map<size_t, std::list<SegmentedBlock>::iterator> getIdxtoIterMap(
    std::list<SegmentedBlock>& segmented_blocks_list) {
  std::unordered_map<size_t, std::list<SegmentedBlock>::iterator> idx_to_iter;
  auto iter = segmented_blocks_list.begin();
  for (uint64_t i = 0; i < segmented_blocks_list.size(); ++i, ++iter) {
    idx_to_iter[i] = iter;
  }
  return idx_to_iter;
}

void resolveNonTensorInputBlocks(PartitionedGraph& segmented_blocks) {
  // get input usage counts and blocks_list
  std::list<SegmentedBlock> segmented_blocks_list(segmented_blocks.cbegin(), segmented_blocks.cend());
  auto usage_counts = getInputUsageCounts(
      segmented_blocks, [](torch::jit::Value* input) -> bool { return !isTensorOrTensorList(input); });
  auto idx_to_iter = getIdxtoIterMap(segmented_blocks_list);

  std::map<int, std::vector<torch::jit::Value*>>
      torch_values_to_fix; // Only need to resolve values generated by tensorrt
  std::set<int> tensorrt_blocks_to_fix; // Need to resolve ALL non-tensor inputs

  // update blocks_list
  std::unordered_set<int> updated_segments;
  for (auto& use : usage_counts) {
    auto use_info = use.second;
    // if the segment that produce this nonTensor value is kTensorRT but consumed in kTorch, inject nodes in the first
    // kTorch segment.
    if (segmented_blocks[use_info.produce_id].target() == SegmentedBlock::kTensorRT && !use_info.torch_use_id.empty()) {
      auto first_torch_id = use_info.torch_use_id.back();
      torch_values_to_fix[first_torch_id].push_back(use.first);
    }
    // kTensorRT segments always need to inject nodes for the nonTensor inputs
    for (auto i : use_info.tensorrt_use_id) {
      tensorrt_blocks_to_fix.insert(i);
    }
  }
  for (auto torch_block_pair : torch_values_to_fix) {
    auto to_inject_blocks =
        segmentBlocksWithSpecifiedInputs(segmented_blocks[torch_block_pair.first], torch_block_pair.second);
    auto next_iter = segmented_blocks_list.erase(idx_to_iter[torch_block_pair.first]);
    segmented_blocks_list.insert(next_iter, to_inject_blocks.begin(), to_inject_blocks.end());
  }

  for (auto i : tensorrt_blocks_to_fix) {
    auto to_inject_blocks = segmentBlocksWithNonTensorInputs(segmented_blocks[i]);
    auto next_iter = segmented_blocks_list.erase(idx_to_iter[i]);
    segmented_blocks_list.insert(next_iter, to_inject_blocks.begin(), to_inject_blocks.end());
  }

  segmented_blocks.clear();
  segmented_blocks.insert(segmented_blocks.begin(), segmented_blocks_list.begin(), segmented_blocks_list.end());
  return;
}

void resolveTensorListInputBlocks(PartitionedGraph& segmented_blocks) {
  // usage_counts is a map with key as non-tensor/tensorlist inputs and value as the idx of segmented block which
  // produces/contains it.
  auto usage_counts =
      getInputUsageCounts(segmented_blocks, [](torch::jit::Value* input) -> bool { return isTensorList(input); });

  // Get idx of the segblock to its iterator mapping
  std::list<SegmentedBlock> segmented_blocks_list(segmented_blocks.cbegin(), segmented_blocks.cend());
  auto idx_to_iter = getIdxtoIterMap(segmented_blocks_list);

  std::unordered_set<int> updated_segments;
  // we need to re-segment TensorRT segments whose inputs are TensorLists
  for (auto& use : usage_counts) {
    auto use_info = use.second;
    // For a particular tensorlist input, traverse through all ids of segmented blocks whose target is TensorRT
    for (auto i : use_info.tensorrt_use_id) {
      if (!updated_segments.count(i)) {
        // tensorlistinput_to_segblock is a mapping from {tensorlist input : segmented block which produced this
        // tensorlist input}
        std::unordered_map<torch::jit::Value*, SegmentedBlock> tensorlistinput_to_segblock;
        for (auto input : segmented_blocks[i].raw_inputs()) {
          if (isTensorList(input)) {
            tensorlistinput_to_segblock[input] = segmented_blocks[usage_counts[input].produce_id];
          }
        }

        // For each tensorlist input in tensorlistinput_to_segblock, get the node which actually uses this input.
        // Once we retrieve the node, we remove it from the current TensorRT segmented_blocks[i]. This node should be
        // added to block that generated/produced (can be obtained via produce_id) this tensorlist input in the first
        // place.
        auto seg_blocks = segmentBlocksWithTensorListInputs(segmented_blocks[i], tensorlistinput_to_segblock);
        auto append_blocks = seg_blocks.first;
        auto trt_block = seg_blocks.second;
        // Remove the current TensorRT seg_block and replace it with new TRT block (non empty) which has the node that
        // uses tensorlist input removed.
        auto next_iter = segmented_blocks_list.erase(idx_to_iter[i]);
        if (trt_block.raw_nodes().size() > 0) {
          segmented_blocks_list.insert(next_iter, trt_block);
        }

        // append blocks' nodes to the producer seg_block
        for (auto append_block : append_blocks) {
          auto input = append_block.first; // corresponds to the tensorlist input
          auto block = append_block.second;
          // append nodes to segmented_blocks_list
          auto producer = idx_to_iter[usage_counts[input].produce_id];
          for (auto n : block.raw_nodes()) {
            producer->cloneNode(n);
          }
        }
        updated_segments.insert(i);
      }
    }
  }
  segmented_blocks.clear();
  segmented_blocks.insert(segmented_blocks.begin(), segmented_blocks_list.begin(), segmented_blocks_list.end());
  return;
}

void resolveNonTensorInputs(PartitionedGraph& segmented_blocks) { // , std::shared_ptr<torch::jit::Graph> g
  // make sure that all inputs should be tensor
  LOG_DEBUG("Resolving nonTensor inputs/outputs of segmented_blocks");
  resolveNonTensorInputBlocks(segmented_blocks);

  // we need to re-segment tensorrt blocks whose inputs are tensorlists (eg: Tensor [] instead of Tensor).
  LOG_DEBUG("Resolving inputs of type TensorList in segmented_blocks");
  resolveTensorListInputBlocks(segmented_blocks);
}

void registerSegmentsOutputs(PartitionedGraph& segmented_blocks, torch::jit::Block* block) {
  // find the corresponding raw values in original global graph for this segmented block's inputs/outputs
  std::set<torch::jit::Value*> input_values;
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
        if (!isTensorOrTensorList(mini_graph_input) && seg_block.target() == SegmentedBlock::kTensorRT)
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
            if (isTensorOrTensorList(node_output))
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

bool should_run_in_trt(torch::jit::Node* n, const std::unordered_set<std::string>& torch_ops) {
  // If the op is not supported by the conversion phase it should run in PyTorch
  if (!conversion::OpSupported(n)) {
    LOG_GRAPH("Node not supported by conversion: " << util::node_info(n));
    return false;
  }

  // If the user specifies the op to run in Torch it should run in PyTorch
  if (torch_ops.find(n->kind().toQualString()) != torch_ops.end()) {
    LOG_GRAPH("Node explicitly set to run in torch: " << util::node_info(n));
    return false;
  }

  // If the user specifies the module containing this op to run in torch it should run in PyTorch
  const auto to_compile_sym = c10::Symbol::attr("to_compile");
  if (n->hasAttribute(to_compile_sym) && n->i(to_compile_sym) == (int64_t) false) {
    LOG_GRAPH("Node is within a module set to run in torch: " << util::node_info(n));
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

PartitionedGraph segment_graph(torch::jit::Block* block, const PartitionInfo& partition_info) {
  auto min_block_size = partition_info.min_block_size;
  std::unordered_set<std::string> forced_fallback_ops(
      partition_info.forced_fallback_operators.begin(), partition_info.forced_fallback_operators.end());

  auto nodes = block->nodes();
  PartitionedGraph segmented_blocks;

  // segment the nodes
  std::vector<torch::jit::Node*> in_prog_trt_blk_nodes, in_prog_pyt_blk_nodes;
  for (const auto n : nodes) {
    // Skip constant nodes as they are resources for both kinds of modules
    if (n->kind() == torch::jit::prim::Constant) {
      continue;
    }

    if (should_run_in_trt(n, forced_fallback_ops)) {
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
    const PartitionInfo& partition_info) {
  LOG_DEBUG(partition_info);
  // segment lowering global graph into blocks
  LOG_DEBUG("Parititioning source module into PyTorch and TensorRT sub blocks");
  PartitionedGraph segmented_blocks = segment_graph(block, partition_info);

  // resolve nonTensor inputs/outputs
  resolveNonTensorInputs(segmented_blocks);

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
