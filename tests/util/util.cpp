#include "core/util/prelude.h"
#include "torch/script.h"

namespace trtorch {
namespace tests {
namespace util {

bool checkRtol(const at::Tensor& diff, const std::vector<at::Tensor> inputs, float threshold) {
  double maxValue = 0.0;
  for (auto& tensor : inputs) {
    maxValue = fmax(tensor.abs().max().item<float>(), maxValue);
  }
  std::cout << "Max Difference: " << diff.abs().max().item<float>() << std::endl;
  std::cout << "Acceptable Threshold: " << threshold << std::endl;
  return diff.abs().max().item<float>() <= threshold * maxValue;
}

bool almostEqual(const at::Tensor& a, const at::Tensor& b, float threshold) {
  LOG_GRAPH(a << std::endl << b << std::endl);
  auto a_float = a.toType(at::kFloat);
  auto b_float = b.toType(at::kFloat);
  return checkRtol(a_float - b_float, {a_float, b_float}, threshold);
}

bool exactlyEqual(const at::Tensor& a, const at::Tensor& b) {
  return (a - b).abs().max().item<float>() == 0.f;
}

} // namespace util
} // namespace tests
} // namespace trtorch
