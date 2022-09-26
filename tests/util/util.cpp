#include "core/util/prelude.h"
#include "torch/script.h"
#include "torch/torch.h"

namespace torch_tensorrt {
namespace tests {
namespace util {

bool cosineSimEqual(const at::Tensor& computed_tensor, const at::Tensor& gt_tensor, float threshold = 0.99f) {
  torch::Tensor cosine_sim = torch::nn::functional::cosine_similarity(
      computed_tensor.flatten(), gt_tensor.flatten(), torch::nn::functional::CosineSimilarityFuncOptions().dim(0));
  std::ostringstream ss;
  ss << computed_tensor << std::endl << gt_tensor << std::endl;
  ss << "cosine similarity" << std::to_string(cosine_sim.item<float>()) << std::endl;
  std::cout << "cosine similarity" << std::to_string(cosine_sim.item<float>()) << std::endl;
  std::cout << "cosine sim" << cosine_sim.item<float>() << std::endl;
  LOG_GRAPH(ss.str());
  LOG_GRAPH(std::string("Cosine Similarity score: ") + std::to_string(cosine_sim.item<float>()));
  LOG_GRAPH(std::string("Acceptable Threshold: ") + std::to_string(threshold));

  return cosine_sim.item<float>() >= threshold;
}

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
