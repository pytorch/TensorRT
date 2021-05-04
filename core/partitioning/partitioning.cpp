#include "partitioning.h"

#include <queue>
#include "core/conversion/conversion.h"
#include "core/partitioning/shape_analysis.h"
#include "torch/csrc/jit/passes/constant_pooling.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"

namespace trtorch {
namespace core {
namespace partitioning {

struct usage_info {
  int produce_id = -1;
  std::vector<int> torch_use_id;
  std::vector<int> tensorrt_use_id;
};

inline bool isTensorOrTensorList(torch::jit::Value* val) {
  return val->type()->isSubtypeOf(torch::jit::TensorType::get()) ||
      val->type()->isSubtypeOf(torch::jit::ListType::ofTensors());
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

std::vector<torch::jit::Node*> getDependencyNodes(std::vector<torch::jit::Value*>& vals) {
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

std::vector<SegmentedBlock> injectNodesForNonTensorInputs(SegmentedBlock& seg_block) {
  // reconstruct segmented_block if this block requires nonTensor input
  std::vector<torch::jit::Value*> nontensor_inputs;
  for (auto input : seg_block.raw_inputs()) {
    if (!isTensorOrTensorList(input)) {
      nontensor_inputs.push_back(input);
    }
  }
  std::vector<torch::jit::Node*> dependency_nodes = getDependencyNodes(nontensor_inputs);

  std::vector<SegmentedBlock> new_seg_blocks;
  // if current block is kTorch or current block is TensorRT and all dependent nodes are also supported, construct only
  // one new block
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
    std::unordered_set<torch::jit::Value*> nontensor_inputs_set(nontensor_inputs.begin(), nontensor_inputs.end());
    std::vector<torch::jit::Node*> tensorrt_nodes, pytorch_nodes;
    bool prev_non_tensor_outputs = false;
    for (auto n : seg_block.raw_nodes()) {
      // it's a kTorch block if it uses the nonTensor input and the nonTensor input is produced in kTorch block
      if (containTargetInputs(n, nontensor_inputs_set) || prev_non_tensor_outputs) {
        if (!tensorrt_nodes.empty()) {
          new_seg_blocks.emplace_back(SegmentedBlock::kTensorRT, tensorrt_nodes);
          tensorrt_nodes.clear();
        }
        pytorch_nodes.push_back(n);
        prev_non_tensor_outputs = containNonTensorOutputs(n);
      } else {
        if (!pytorch_nodes.empty()) {
          new_seg_blocks.emplace_back(SegmentedBlock::kTorch, pytorch_nodes);
          pytorch_nodes.clear();
        }
        tensorrt_nodes.push_back(n);
      }
    }
    if (!tensorrt_nodes.empty()) {
      new_seg_blocks.emplace_back(SegmentedBlock::kTensorRT, tensorrt_nodes);
    } else {
      new_seg_blocks.emplace_back(SegmentedBlock::kTorch, pytorch_nodes);
    }
  }
  return std::move(new_seg_blocks);
}

void resolveNonTensorInputs(PartitionedGraph& segmented_blocks) {
  // for NonTensor inputs in TensorRT segments, count the usages on Torch segments and TensorRT segments
  std::unordered_map<torch::jit::Value*, usage_info> usage_counts;
  for (int i = segmented_blocks.size() - 1; i >= 0; --i) {
    for (auto input : segmented_blocks[i].raw_inputs()) {
      if (!isTensorOrTensorList(input)) {
        segmented_blocks[i].target() == SegmentedBlock::kTorch ? usage_counts[input].torch_use_id.push_back(i)
                                                               : usage_counts[input].tensorrt_use_id.push_back(i);
      }
    }
    for (auto& use : usage_counts) {
      if (segmented_blocks[i].contain_raw_value(use.first)) {
        use.second.produce_id = i;
      }
    }
  }
  std::unordered_set<int> updated_segments;
  for (auto& use : usage_counts) {
    auto use_info = use.second;
    // if the segment that produce this nonTensor value is kTensorRT but consumed in kTorch, inject nodes in the first
    // kTorch segments
    if (segmented_blocks[use_info.produce_id].target() == SegmentedBlock::kTensorRT && !use_info.torch_use_id.empty()) {
      int first_torch_id = use_info.torch_use_id.front();
      if (!updated_segments.count(first_torch_id)) {
        auto to_inject_blocks = injectNodesForNonTensorInputs(segmented_blocks[first_torch_id]);
        segmented_blocks.erase(segmented_blocks.begin() + first_torch_id);
        segmented_blocks.insert(
            segmented_blocks.begin() + first_torch_id, to_inject_blocks.begin(), to_inject_blocks.end());
        updated_segments.insert(first_torch_id);
      }
    } else {
      // KTensorRT segments always need to inject nodes for the nonTensor inputs
      for (int i : use_info.tensorrt_use_id) {
        if (!updated_segments.count(i)) {
          auto to_inject_blocks = injectNodesForNonTensorInputs(segmented_blocks[i]);
          segmented_blocks.erase(segmented_blocks.begin() + i);
          segmented_blocks.insert(segmented_blocks.begin() + i, to_inject_blocks.begin(), to_inject_blocks.end());
          updated_segments.insert(i);
        }
      }
    }
  }
  return;
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

std::vector<SegmentedBlock> segment_graph(torch::jit::Block* block, const PartitionInfo& partition_info) {
  auto min_block_size = partition_info.min_block_size;
  std::unordered_set<std::string> forced_fallback_operators(
      partition_info.forced_fallback_operators.begin(), partition_info.forced_fallback_operators.end());

  auto nodes = block->nodes();
  std::vector<SegmentedBlock> segmented_blocks;

  // segment the nodes
  std::vector<torch::jit::Node*> tensorrt_nodes, pytorch_nodes;
  for (const auto n : nodes) {
    if (n->kind() == torch::jit::prim::Constant) {
      continue;
    }

    std::string node_string(n->kind().toQualString());
    if (conversion::OpSupported(n) && !forced_fallback_operators.count(node_string)) {
      tensorrt_nodes.push_back(n);
      if (tensorrt_nodes.size() >= min_block_size && !pytorch_nodes.empty()) {
        segmented_blocks.emplace_back(SegmentedBlock::kTorch, pytorch_nodes);
        pytorch_nodes.clear();
      }
    } else {
      if (tensorrt_nodes.size() >= min_block_size) {
        segmented_blocks.emplace_back(SegmentedBlock::kTensorRT, tensorrt_nodes);
      } else {
        pytorch_nodes.insert(pytorch_nodes.end(), tensorrt_nodes.begin(), tensorrt_nodes.end());
      }
      tensorrt_nodes.clear();
      // if there is a prim::If then this if node will be encapsulated in a SegmentedBlock
      // we shouldn't inject node for this block in dependency analysis process
      if (n->kind() == torch::jit::prim::If) {
        if (!pytorch_nodes.empty()) {
          segmented_blocks.emplace_back(SegmentedBlock::kTorch, pytorch_nodes);
          pytorch_nodes.clear();
        }
        segmented_blocks.emplace_back(SegmentedBlock::kTorch, std::vector<torch::jit::Node*>{n});
        continue;
      }
      pytorch_nodes.push_back(n);
    }
  }

  // if there is any kTorch nodes left, then either the last nodes are kTorch or last nodes are kTensorRT but num <
  // min_block_size
  if (!pytorch_nodes.empty()) {
    pytorch_nodes.insert(pytorch_nodes.end(), tensorrt_nodes.begin(), tensorrt_nodes.end());
    segmented_blocks.emplace_back(SegmentedBlock::kTorch, pytorch_nodes);
  } else {
    segmented_blocks.emplace_back(SegmentedBlock::kTensorRT, tensorrt_nodes);
  }

  return std::move(segmented_blocks);
}

std::vector<SegmentedBlock> Partition(
    torch::jit::Block* block,
    std::unordered_map<torch::jit::Value*, torch::jit::IValue>& input_ivalues_map,
    const PartitionInfo& partition_info) {
  LOG_DEBUG(partition_info);
  // segment lowering global graph into blocks
  std::vector<SegmentedBlock> segmented_blocks = segment_graph(block, partition_info);

  // resolve nonTensor inputs/outputs
  resolveNonTensorInputs(segmented_blocks);

  // register input/output torch::jit::Value for segmented graphs
  registerSegmentsOutputs(segmented_blocks, block);

  // run shape analysis on each segmented block
  runShapeAnalysis(segmented_blocks, input_ivalues_map);

  return segmented_blocks;
}

} // namespace partitioning
} // namespace core
} // namespace trtorch
