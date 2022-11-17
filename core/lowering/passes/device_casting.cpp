#include "torch/csrc/jit/ir/constants.h"
#include "torch/csrc/jit/passes/subgraph_rewrite.h"

#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace lowering {
namespace passes {

void UnpackAndCastMaskedFill(std::shared_ptr<torch::jit::Graph>& graph, std::string target_device_name) {
  std::string masked_fill_pattern = R"IR(
    graph(%self, %mask, %value):
      %out: Tensor = aten::masked_fill_(%self, %mask, %value)
      return (%out))IR";

  // Calls to masked_fill_ often utilize CPU tensors, and as such
  // should be moved to gpu to avoid device mismatch errors

  // Separate string into portions to insert device name
  std::string clean_pattern_part_1 = R"IR(
    graph(%self, %mask, %value):
      %device: Device = prim::Constant[value=")IR";

  std::string clean_pattern_part_2 = R"IR("]()
      %dtype: NoneType = prim::Constant()
      %false: bool = prim::Constant[value=0]()
      %mask_cuda: Tensor = aten::to(%mask, %device, %dtype, %false, %false)
      %self_cuda: Tensor = aten::to(%self, %device, %dtype, %false, %false)
      %out: Tensor = aten::masked_fill(%self_cuda, %mask_cuda, %value)
      return (%out))IR";

  auto unpacked_pattern = clean_pattern_part_1 + target_device_name + clean_pattern_part_2;

  torch::jit::SubgraphRewriter masked_fill_rewriter;
  masked_fill_rewriter.RegisterRewritePattern(masked_fill_pattern, unpacked_pattern);
  masked_fill_rewriter.runOnGraph(graph);
  LOG_GRAPH("After unpack and cast masked_fill_: " << *graph);
}

void UnpackAndCastNumToTensor(std::shared_ptr<torch::jit::Graph>& graph, std::string target_device_name) {
  std::string num_to_tensor_cast_pattern = R"IR(
    graph(%1: Scalar):
      %2: Tensor = prim::NumToTensor(%1)
      return (%2))IR";

  // 0D Tensors are initialized on cpu, and need to be moved to gpu
  // to avoid device mismatch issues

  // Separate string into portions to insert device name
  std::string clean_pattern_part_1 = R"IR(
    graph(%1: Scalar):
      %2: Tensor = prim::NumToTensor(%1)
      %device: Device = prim::Constant[value=")IR";

  std::string clean_pattern_part_2 = R"IR("]()
      %dtype: NoneType = prim::Constant()
      %false: bool = prim::Constant[value=0]()
      %3: Tensor = aten::to(%2, %device, %dtype, %false, %false)
      return (%3))IR";

  auto num_to_tensor_clean_pattern = clean_pattern_part_1 + target_device_name + clean_pattern_part_2;

  torch::jit::SubgraphRewriter num_to_tensor_cast_rewriter;
  num_to_tensor_cast_rewriter.RegisterRewritePattern(num_to_tensor_cast_pattern, num_to_tensor_clean_pattern);
  num_to_tensor_cast_rewriter.runOnGraph(graph);

  LOG_GRAPH("After unpack and cast NumToTensor: " << *graph);
}

void UnpackAndCastFull(std::shared_ptr<torch::jit::Graph>& graph, std::string target_device_name) {
  std::string full_cast_pattern = R"IR(
    graph(%1, %2, %3, %4, %5, %6):
      %out: Tensor = aten::full(%1, %2, %3, %4, %5, %6)
      return (%out))IR";

  // Tensors created via aten::full are initialized on cpu, and need to be casted to gpu
  // to avoid device mismatch issues

  // Separate string into portions to insert device name
  std::string clean_pattern_part_1 = R"IR(
    graph(%1, %2, %3, %4, %5, %6):
      %device: Device = prim::Constant[value=")IR";

  std::string clean_pattern_part_2 = R"IR("]()
      %out: Tensor = aten::full(%1, %2, %3, %4, %device, %6)
      return (%out))IR";

  auto full_clean_pattern = clean_pattern_part_1 + target_device_name + clean_pattern_part_2;

  torch::jit::SubgraphRewriter full_cast_rewriter;
  full_cast_rewriter.RegisterRewritePattern(full_cast_pattern, full_clean_pattern);
  full_cast_rewriter.runOnGraph(graph);

  LOG_GRAPH("After unpack and cast full: " << *graph);
}

void ReplaceScalarImplicit(std::shared_ptr<torch::jit::Graph>& graph) {
  std::string scalar_implicit_cast_pattern = R"IR(
    graph(%1: Tensor):
      %2: Scalar = aten::ScalarImplicit(%1)
      return (%2))IR";

  // ScalarImplicit can only unpack 0D tensors, whereas Tensors operated on by
  // TensorRT are padded to 1 dimension. aten::item() resolves this conflict
  std::string scalar_implicit_clean_pattern = R"IR(
    graph(%1: Tensor):
      %2: Scalar = aten::item(%1)
      return (%2))IR";

  torch::jit::SubgraphRewriter scalar_implicit_cast_rewriter;
  scalar_implicit_cast_rewriter.RegisterRewritePattern(scalar_implicit_cast_pattern, scalar_implicit_clean_pattern);
  scalar_implicit_cast_rewriter.runOnGraph(graph);

  LOG_GRAPH("After unpack and cast full: " << *graph);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
