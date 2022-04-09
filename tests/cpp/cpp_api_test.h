#include <utility>
#include "c10/cuda/CUDACachingAllocator.h"
#include "cuda_runtime_api.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/script.h"
#include "torch_tensorrt/torch_tensorrt.h"

using PathAndInput = std::tuple<std::string, std::vector<std::vector<int64_t>>, std::vector<c10::ScalarType>, float>;

class CppAPITests : public testing::TestWithParam<PathAndInput> {
 public:
  void SetUp() override {
    PathAndInput params = GetParam();
    std::string path = std::get<0>(params);
    try {
      // Deserialize the ScriptModule from a file using torch::jit::load().
      mod = torch::jit::load(path);
    } catch (const c10::Error& e) {
      std::cerr << "error loading the model\n";
      ASSERT_TRUE(false);
    }
    input_shapes = std::get<1>(params);
    input_types = std::get<2>(params);
    threshold = std::get<3>(params);
  }

  void TearDown() {
    cudaDeviceSynchronize();
    c10::cuda::CUDACachingAllocator::emptyCache();
  }

 protected:
  torch::jit::script::Module mod;
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<c10::ScalarType> input_types;
  float threshold;
};
