#include "util.h"
#include <string>
#include "core/util/prelude.h"
#include "gtest/gtest.h"
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

void pointwise_test_helper(
    std::string graph_ir,
    bool singleInput,
    bool dynamicInput,
    std::vector<int64_t> shape1,
    std::vector<int64_t> shape2,
    bool negative_input,
    at::ScalarType type1,
    at::ScalarType type2) {
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph_ir, g.get());

  // singleInput case is enabled when elementwise operation is performed
  // with an input and a constant embedded in graph
  std::vector<at::Tensor> torch_inputs;
  int first_min = negative_input ? -5 : 1;
  int first_max = 5;
  int second_min = 1;
  int second_max = 5;
  if (type1 == at::kBool) {
    first_min = 0;
    first_max = 1;
  }
  if (type2 == at::kBool) {
    second_min = 0;
    second_max = 1;
  }
  torch_inputs.push_back(at::randint(first_min, first_max, shape1, at::TensorOptions(at::kCUDA).dtype(type1)));
  if (!singleInput) {
    torch_inputs.push_back(at::randint(second_min, second_max, shape2, at::TensorOptions(at::kCUDA).dtype(type2)));
  }

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, torch_inputs);

  std::vector<at::Tensor> trt_inputs;
  for (auto in : torch_inputs) {
    trt_inputs.push_back(at::clone(in));
  }

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  std::vector<at::Tensor> trt_results;
  if (dynamicInput) {
    trt_results = torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, trt_inputs);
  } else {
    trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, trt_inputs);
  }

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

} // namespace util
} // namespace tests
} // namespace torch_tensorrt
