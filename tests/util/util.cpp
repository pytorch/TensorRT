#include "core/util/prelude.h"
#include "torch/script.h"

namespace torch_tensorrt {
namespace tests {
namespace util {

bool almostEqual(const at::Tensor& a, const at::Tensor& b, float threshold, float atol = 1e-8, float rtol = 1e-5) {
  LOG_GRAPH(a << std::endl << b << std::endl);
  auto a_float = a.toType(at::kFloat);
  auto b_float = b.toType(at::kFloat);

  auto diff = a_float - b_float;
  auto result = diff.abs().max().item<float>() - (atol + rtol * b.abs().max().item<float>());

  std::cout << "Max Difference: " << result << std::endl;
  std::cout << "Acceptable Threshold: " << threshold << std::endl;

  return result <= threshold;
}

bool exactlyEqual(const at::Tensor& a, const at::Tensor& b) {
  LOG_GRAPH(a << std::endl << b << std::endl);
  std::cout << "Max Difference: " << (a - b).abs().max().item<float>() << std::endl;

  return (a - b).abs().max().item<float>() == 0.f;
}

} // namespace util
} // namespace tests
} // namespace torch_tensorrt
