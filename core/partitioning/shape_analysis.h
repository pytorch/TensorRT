#include "SegmentedBlock.h"

namespace trtorch {
namespace core {
namespace partitioning {

std::vector<torch::jit::IValue> generateRandomInputs(std::vector<conversion::InputRange>& input_ranges);

void getSegmentsOutputByRunning(
    SegmentedBlock& seg_block,
    std::unordered_map<torch::jit::Value*, torch::jit::IValue>& ivalues_maps);

} // namespace partitioning
} // namespace core
} // namespace trtorch