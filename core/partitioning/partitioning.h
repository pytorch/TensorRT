#pragma once

#include <iostream>
#include <vector>

#include "torch/csrc/jit/ir/ir.h"

#include "core/ir/ir.h"
#include "core/partitioning/partitioningctx/PartitioningCtx.h"
#include "core/partitioning/partitioninginfo/PartitioningInfo.h"
#include "core/partitioning/segmentedblock/SegmentedBlock.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace partitioning {

typedef std::unordered_map<const torch::jit::Value*, torch::jit::IValue> ExampleIValues;

typedef std::pair<std::shared_ptr<torch::jit::Graph>, std::unordered_map<torch::jit::Value*, torch::jit::Value*>>
    GraphAndMapping;

ExampleIValues generateRandomInputs(ir::CollectionInputSpecMap& input_ranges, ir::CollectionTypeMap& input_types);

void runShapeAnalysis(PartitioningCtx* ctx, torch::jit::Block* block, ExampleIValues& ivalues_maps);

void segment_graph(PartitioningCtx* ctx, torch::jit::Block* block);

GraphAndMapping Stitch(PartitioningCtx* ctx, torch::jit::Block* block);

void Partition(PartitioningCtx* ctx, ExampleIValues& example_tensor_map);

} // namespace partitioning
} // namespace core
} // namespace torch_tensorrt
