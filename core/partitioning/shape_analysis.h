#include "core/ir/ir.h"
#include "core/partitioning/SegmentedBlock.h"
#include "torch/csrc/jit/ir/ir.h"

namespace torch_tensorrt {
namespace core {
namespace partitioning {

std::unordered_map<const torch::jit::Value*, torch::jit::IValue> generateRandomInputs(
    std::unordered_map<const torch::jit::Value*, std::vector<ir::Input>>& input_ranges,
    std::unordered_map<const torch::jit::Value*, std::vector<c10::optional<at::ScalarType>>>& input_types);

void runShapeAnalysis(
    std::vector<SegmentedBlock>& segmented_blocks,
    std::unordered_map<const torch::jit::Value*, torch::jit::IValue>& ivalues_maps,
    const PartitionInfo& partition_info);

} // namespace partitioning
} // namespace core
} // namespace torch_tensorrt
