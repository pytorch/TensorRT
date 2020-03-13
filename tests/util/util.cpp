#include "core/util/prelude.h"
#include "torch/script.h"

namespace trtorch {
namespace tests {
namespace util {

bool checkRtol(const at::Tensor& diff, const std::vector<at::Tensor> inputs) {
    double maxValue = 0.0;
    for (auto& tensor : inputs) {
        maxValue = fmax(tensor.abs().max().item<float>(), maxValue);
    }
    std::cout << "Max Difference: " << diff.abs().max().item<float>() << std::endl;
    return diff.abs().max().item<float>() <= 2e-6 * maxValue;
}

bool almostEqual(const at::Tensor& a, const at::Tensor& b) {
    LOG_DEBUG(a << std::endl << b << std::endl);
    return checkRtol(a - b, {a, b});
}

bool exactlyEqual(const at::Tensor& a, const at::Tensor& b) {
  return (a - b).abs().max().item<float>() == 0.f;
}

} // namespace util
} // namespace tests
} // namespace trtorch
