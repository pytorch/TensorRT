#pragma once

#include <iostream>
#include <vector>

#include "torch/csrc/jit/ir/ir.h"

#include "core/ir/ir.h"
#include "core/partitioning/partitioningctx/PartitioningCtx.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace partitioning {

typedef std::unordered_map<const torch::jit::Value*, torch::jit::IValue> ExampleIValues;

typedef std::pair<std::shared_ptr<torch::jit::Graph>, std::unordered_map<torch::jit::Value*, torch::jit::Value*>>
    GraphAndMapping;

// Set of schemas allowed to be executed in Torch, even with require_full_compilation=true,
// as necessary for returning collections of Tensors or other complex constructs, and for
// processing inputs to TRT engines
const std::unordered_set<std::string> CollectionSchemas = {
    "prim::Constant",
    "aten::__getitem__",
    "prim::ListConstruct",
    "prim::ListUnpack",
    "prim::TupleIndex",
    "prim::TupleConstruct",
    "prim::TupleUnpack",
};

ExampleIValues generateRandomInputs(
    ir::CollectionInputSpecMap& input_ranges,
    ir::CollectionTypeMap& input_types,
    const ir::ShapeMode& shape_mode = ir::ShapeMode::kOPT);

void populateInputIValues(PartitioningCtx* ctx);

void runShapeAnalysis(
    PartitioningCtx* ctx,
    torch::jit::Block* block,
    ExampleIValues& ivalues_maps,
    const ir::ShapeMode& shape_mode);

void segmentGraph(PartitioningCtx* ctx, torch::jit::Block* block);

GraphAndMapping stitch(PartitioningCtx* ctx, torch::jit::Block* block);

void partition(PartitioningCtx* ctx);

} // namespace partitioning
} // namespace core
} // namespace torch_tensorrt
