#include "core/ir/ir.h"
#include "core/partitioning/SegmentedBlock.h"
#include "torch/csrc/jit/ir/ir.h"

namespace trtorch {
namespace core {
namespace partitioning {

std::unordered_map<torch::jit::Value*, torch::jit::IValue> generateRandomInputs(
    std::unordered_map<torch::jit::Value*, ir::InputRange>& input_ranges);

void runShapeAnalysis(
    std::vector<SegmentedBlock>& segmented_blocks,
    std::unordered_map<torch::jit::Value*, torch::jit::IValue>& ivalues_maps);

} // namespace partitioning
} // namespace core
} // namespace trtorch
