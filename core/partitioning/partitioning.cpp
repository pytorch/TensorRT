#include "partitioning.h"

#include <queue>
#include "core/conversion/conversion.h"
#include "core/partitioning/shape_analysis.h"
#include "torch/csrc/jit/passes/constant_pooling.h"

namespace trtorch {
namespace core {
namespace partitioning {

inline bool isTensorOrTensorList(torch::jit::Value* val) {
  return val->type()->isSubtypeOf(torch::jit::TensorType::get()) ||
      val->type()->isSubtypeOf(torch::jit::ListType::ofTensors());
}

struct usage_info {
  int produce_id = -1;
  std::vector<int> torch_use_id;
  std::vector<int> tensorrt_use_id;
};

std::vector<torch::jit::Node*> getDependencyNodes(std::vector<torch::jit::Value*>& vals) {
  // using bfs to get the DAG dependency nodes for input value
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

SegmentedBlock injectNodesForNonTensorInputs(SegmentedBlock& seg_block) {
  // reconstruct segmented_block if this block requires nonTensor input
  std::vector<torch::jit::Value*> nontensor_inputs;
  for (auto input : seg_block.raw_inputs()) {
    if (!isTensorOrTensorList(input)) {
      nontensor_inputs.push_back(input);
    }
  }
  std::vector<torch::jit::Node*> new_block_nodes = getDependencyNodes(nontensor_inputs);
  new_block_nodes.insert(new_block_nodes.end(), seg_block.raw_nodes().begin(), seg_block.raw_nodes().end());
  return std::move(SegmentedBlock(seg_block.target(), new_block_nodes));
}

void resolveNonTensorInputs(PartitionedGraph& segmented_blocks, std::shared_ptr<torch::jit::Graph> g) {
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
        auto new_torch_block = injectNodesForNonTensorInputs(segmented_blocks[first_torch_id]);
        segmented_blocks[first_torch_id] = new_torch_block;
        updated_segments.insert(first_torch_id);
      }
    } else {
      // KTensorRT segments always need to inject nodes for the nonTensor inputs
      for (int i : use_info.tensorrt_use_id) {
        if (!updated_segments.count(i)) {
          auto new_seg_block = injectNodesForNonTensorInputs(segmented_blocks[i]);
          segmented_blocks[i] = new_seg_block;
          updated_segments.insert(i);
        }
      }
    }
  }
  return;
}

void registerSegmentsOutputs(PartitionedGraph& segmented_blocks, std::shared_ptr<torch::jit::Graph> g) {
  // find the corresponding raw values in original global graph for this segmented block's inputs/outputs
  std::set<torch::jit::Value*> input_values;
  for (auto& seg_block : segmented_blocks) {
    for (auto& input : seg_block.raw_inputs()) {
      input_values.insert(input);
    }
  }

  for (auto& graph_output : g->outputs()) {
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
  // erase segments which still have no output
  segmented_blocks.erase(
      std::remove_if(
          segmented_blocks.begin(),
          segmented_blocks.end(),
          [](SegmentedBlock& seg_block) { return seg_block.raw_outputs().empty(); }),
      segmented_blocks.end());

  return;
}

std::vector<SegmentedBlock> segment_graph(
    std::shared_ptr<torch::jit::Graph> g,
    const PartitionInfo& partition_info) {
  auto min_block_size = partition_info.min_block_size;
  std::unordered_set<std::string> forced_fallback_operators(
      partition_info.forced_fallback_operators.begin(), partition_info.forced_fallback_operators.end());

  auto nodes = g->block()->nodes();
  std::vector<SegmentedBlock> segmented_blocks;

  // segment the nodes
  std::vector<torch::jit::Node*> tensorrt_nodes, pytorch_nodes;
  for (const auto n : nodes) {
    if (n->kind() == torch::jit::prim::Constant)
      continue;

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
    std::shared_ptr<torch::jit::Graph> g,
    std::vector<ir::InputRange>& input_ranges,
    const PartitionInfo& partition_info) {
  // segment lowering global graph into blocks
  std::vector<SegmentedBlock> segmented_blocks = segment_graph(g, partition_info);

  // resolve nonTensor inputs/outputs
  resolveNonTensorInputs(segmented_blocks, g);

  // register input/output torch::jit::Value for segmented graphs
  registerSegmentsOutputs(segmented_blocks, g);

  // store the mapping from lowering graph torch::jit::Value => torch::jit::IValue that we get by running segments
  std::unordered_map<torch::jit::Value*, torch::jit::IValue> ivalues_maps;
  std::vector<torch::jit::IValue> random_inputs = generateRandomInputs(input_ranges);
  for (size_t i = 0; i < g->inputs().size(); ++i) {
    ivalues_maps[g->inputs()[i]] = random_inputs[i];
  }

  // register every segment's input shape, and it's running output IValues
  for (auto& seg_block : segmented_blocks) {
    torch::jit::ConstantPooling(seg_block.g());
    getSegmentsOutputByRunning(seg_block, ivalues_maps);
  }

  return segmented_blocks;
}

} // namespace partitioning
} // namespace core
} // namespace trtorch
