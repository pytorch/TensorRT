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

bool almost_equal(const at::Tensor& a, const at::Tensor& b, float atol, float rtol) {
  auto a_float = a.toType(at::kFloat);
  auto b_float = b.toType(at::kFloat);

  auto diff = a_float - b_float;
  auto result = diff.abs().max().item<float>();
  auto threshold = atol + (rtol * b.abs().max().item<float>());

  std::cout << "Max Difference: " << result << std::endl;
  std::cout << "Acceptable Threshold: " << threshold << std::endl;

  return result <= threshold;
}

} // namespace accuracy
} // namespace torchtrtc
