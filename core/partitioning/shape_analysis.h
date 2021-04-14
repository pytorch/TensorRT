#include "core/ir/ir.h"
#include "core/partitioning/SegmentedBlock.h"

namespace trtorch {
namespace core {
namespace partitioning {

std::vector<torch::jit::IValue> generateRandomInputs(std::vector<ir::InputRange>& input_ranges);

void getSegmentsOutputByRunning(
    SegmentedBlock& seg_block,
    std::unordered_map<torch::jit::Value*, torch::jit::IValue>& ivalues_maps);

} // namespace partitioning
} // namespace core
} // namespace trtorch