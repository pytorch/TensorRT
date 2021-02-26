#include "partitioning.h"
#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace partitioning {


std::vector<SegmentedBlock> segment_graph(std::shared_ptr<torch::jit::Graph> g) {
  std::vector<SegmentedBlock> segmented_blocks;

  auto nodes = g->block()->nodes();

  for (const auto n : nodes) {
    if (n->kind() == torch::jit::prim::Constant) continue;
    auto block_target = conversion::OpSupported(n) ? SegmentedBlock::kTensorRT : SegmentedBlock::kTorch;

    if (segmented_blocks.empty() || block_target != segmented_blocks.back().target) {
      SegmentedBlock cur_block(block_target);
      cur_block.appendNode(n);
      segmented_blocks.push_back(cur_block);
    } else {
        segmented_blocks.back().appendNode(n);
    }
  }

  for (auto &seg_block : segmented_blocks) {
    seg_block.registerOutput();
  }

  return segmented_blocks;
}

}
}
}


