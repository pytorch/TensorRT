#include "util.h"
#include "core/util/prelude.h"
#include "torch/script.h"
#include "torch/torch.h"

namespace torch_tensorrt {
namespace tests {
namespace util {

bool almostEqual(const at::Tensor& computed_tensor, const at::Tensor& gt_tensor, float atol, float rtol) {
  std::ostringstream ss;
  ss << computed_tensor << std::endl << gt_tensor << std::endl;
  ss << " atol: " << atol << " rtol: " << rtol << std::endl;

  LOG_GRAPH(ss.str());
  auto computed_tensor_float = computed_tensor.toType(at::kFloat);
  auto gt_tensor_float = gt_tensor.toType(at::kFloat);

  auto diff = computed_tensor_float - gt_tensor_float;
  auto result = diff.abs().max().item<float>();
  auto threshold = atol + (rtol * gt_tensor.abs().max().item<float>());

  LOG_DEBUG(std::string("Max Difference: ") + std::to_string(result));
  LOG_DEBUG(std::string("Acceptable Threshold: ") + std::to_string(threshold));

  return result <= threshold;
}

bool cosineSimEqual(const at::Tensor& computed_tensor, const at::Tensor& gt_tensor, float threshold) {
  torch::Tensor cosine_sim = torch::nn::functional::cosine_similarity(
      computed_tensor.flatten(), gt_tensor.flatten(), torch::nn::functional::CosineSimilarityFuncOptions().dim(0));
  std::ostringstream ss;
  ss << computed_tensor << std::endl << gt_tensor << std::endl;
  LOG_DEBUG(ss.str());
  if (computed_tensor.sum().item<float>() == 0.f || gt_tensor.sum().item<float>() == 0.f) {
    return almostEqual(computed_tensor, gt_tensor);
  } else {
    LOG_DEBUG(std::string("Cosine Similarity score: ") + std::to_string(cosine_sim.item<float>()));
    LOG_DEBUG(std::string("Acceptable Threshold: ") + std::to_string(threshold));
    return cosine_sim.item<float>() >= threshold;
  }
}

bool exactlyEqual(const at::Tensor& computed_tensor, const at::Tensor& gt_tensor) {
  LOG_DEBUG(computed_tensor << std::endl << gt_tensor << std::endl);
  std::cout << "Max Difference: " << (computed_tensor - gt_tensor).abs().max().item<float>() << std::endl;

  return (computed_tensor - gt_tensor).abs().max().item<float>() == 0.f;
}

} // namespace util
} // namespace tests
} // namespace torch_tensorrt
