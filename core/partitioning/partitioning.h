#pragma once

#include <vector>

#include "core/conversion/conversion.h"
#include "core/conversion/evaluators/eval_util.h"
#include "core/partitioning/SegmentedBlock.h"
#include "core/util/prelude.h"
#include "torch/csrc/jit/ir/ir.h"

namespace trtorch {
namespace core {
namespace partitioning {

//std::vector<SegmentedBlock> Partition(
//    std::shared_ptr<torch::jit::Graph> g,
//    std::vector<conversion::InputRange>& input_ranges,
//    const conversion::TorchFallback& fallback_info);


std::vector<SegmentedBlock> Partition(
    torch::jit::Block* block,
    std::vector<conversion::InputRange>& input_ranges,
    const conversion::TorchFallback& fallback_info);



} // namespace partitioning
} // namespace core
} // namespace trtorch