#include "core/ir/ir.h"
#include "core/partitioning/SegmentedBlock.h"
#include "torch/csrc/jit/ir/ir.h"

namespace trtorch {
namespace core {
namespace partitioning {

std::vector<torch::jit::IValue> generateRandomInputs(std::vector<ir::Input>& inputs);

void runShapeAnalysis(
    std::vector<SegmentedBlock>& segmented_blocks,
    std::vector<ir::Input>& inputs,
    std::shared_ptr<torch::jit::Graph> g);

} // namespace partitioning
} // namespace core
} // namespace trtorch