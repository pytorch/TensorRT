#include "core/util/prelude.h"
#include "torch/script.h"

namespace torch_tensorrt {
namespace tests {
namespace util {

bool almostEqual(const at::Tensor& computed_tensor, const at::Tensor& gt_tensor, float atol = 1e-8, float rtol = 1e-5) {
  std::ostringstream ss;
  ss << computed_tensor << std::endl << gt_tensor << std::endl;
  ss << " atol: " << atol << " rtol: " << rtol << std::endl;

  LOG_GRAPH(ss.str());
  auto computed_tensor_float = computed_tensor.toType(at::kFloat);
  auto gt_tensor_float = gt_tensor.toType(at::kFloat);

  auto diff = computed_tensor_float - gt_tensor_float;
  auto result = diff.abs().max().item<float>();
  auto threshold = atol + (rtol * gt_tensor.abs().max().item<float>());

  LOG_GRAPH(std::string("Max Difference: ") + std::to_string(result));
  LOG_GRAPH(std::string("Acceptable Threshold: ") + std::to_string(threshold));

  return result <= threshold;
}

bool exactlyEqual(const at::Tensor& computed_tensor, const at::Tensor& gt_tensor) {
  LOG_GRAPH(computed_tensor << std::endl << gt_tensor << std::endl);
  std::cout << "Max Difference: " << (computed_tensor - gt_tensor).abs().max().item<float>() << std::endl;

  return (computed_tensor - gt_tensor).abs().max().item<float>() == 0.f;
}

} // namespace util
} // namespace tests
} // namespace torch_tensorrt
