#include "core/ir/ir.h"
#include "core/partitioning/SegmentedBlock.h"
#include "torch/csrc/jit/ir/ir.h"

namespace trtorch {
namespace core {
namespace partitioning {

std::unordered_map<const torch::jit::Value*, torch::jit::IValue> generateRandomInputs(
    std::unordered_map<const torch::jit::Value*, ir::Input>& input_ranges,
    std::unordered_map<const torch::jit::Value*, c10::optional<at::ScalarType>>& input_types);

void runShapeAnalysis(
    std::vector<SegmentedBlock>& segmented_blocks,
    std::unordered_map<const torch::jit::Value*, torch::jit::IValue>& ivalues_maps,
    const PartitionInfo& partition_info);

} // namespace partitioning
} // namespace core
} // namespace trtorch
