#include "accuracy.h"

#include "torch_tensorrt/logging.h"
#include "torch_tensorrt/torch_tensorrt.h"

namespace torchtrtc {
namespace accuracy {

bool check_rtol(const at::Tensor& diff, const std::vector<at::Tensor> inputs, float threshold) {
  double maxValue = 0.0;
  for (auto& tensor : inputs) {
    maxValue = fmax(tensor.abs().max().item<float>(), maxValue);
  }
  torchtrt::logging::log(
      torchtrt::logging::Level::kDEBUG,
      std::string("Max Difference: ") + std::to_string(diff.abs().max().item<float>()));
  torchtrt::logging::log(
      torchtrt::logging::Level::kDEBUG, std::string("Acceptable Threshold: ") + std::to_string(threshold));
  return diff.abs().max().item<float>() <= threshold * maxValue;
}

bool almost_equal(
    const at::Tensor& computed_tensor,
    const at::Tensor& gt_tensor, // gt_tensor : Ground Truth Tensor
    float atol,
    float rtol) {
  auto computed_tensor_float = computed_tensor.toType(at::kFloat);
  auto gt_tensor_float = gt_tensor.toType(at::kFloat);

  auto diff = computed_tensor_float - gt_tensor_float;
  auto result = diff.abs().max().item<float>();
  auto threshold = atol + (rtol * gt_tensor.abs().max().item<float>());

  torchtrt::logging::log(torchtrt::logging::Level::kDEBUG, std::string("Max Difference: ") + std::to_string(result));
  torchtrt::logging::log(
      torchtrt::logging::Level::kDEBUG, std::string("Acceptable Threshold: ") + std::to_string(threshold));

  return result <= threshold;
}

} // namespace accuracy
} // namespace torchtrtc
