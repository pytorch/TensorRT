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

ExampleIValues generateRandomInputs(ir::CollectionInputSpecMap& input_ranges, ir::CollectionTypeMap& input_types);

void runShapeAnalysis(PartitioningCtx* ctx, ExampleIValues& ivalues_maps);

void segment_graph(PartitioningCtx* ctx, torch::jit::Block* block);

PartitionedGraph Partition(PartitioningCtx* ctx, torch::jit::Block* block, ExampleIValues& example_tensor_map);

} // namespace partitioning
} // namespace core
} // namespace torch_tensorrt
