#include <utility>
#include "c10/cuda/CUDACachingAllocator.h"
#include "cuda_runtime_api.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/script.h"
#include "trtorch/trtorch.h"

using PathAndInSize = std::tuple<std::string, std::vector<std::vector<int64_t>>, float>;

class CppAPITests : public testing::TestWithParam<PathAndInSize> {
 public:
  void SetUp() override {
    PathAndInSize params = GetParam();
    std::string path = std::get<0>(params);
    try {
      // Deserialize the ScriptModule from a file using torch::jit::load().
      mod = torch::jit::load(path);
    } catch (const c10::Error& e) {
      std::cerr << "error loading the model\n";
      return;
    }
    input_shapes = std::get<1>(params);
    threshold = std::get<2>(params);
  }

  void TearDown() {
    cudaDeviceSynchronize();
    c10::cuda::CUDACachingAllocator::emptyCache();
  }

 protected:
  torch::jit::script::Module mod;
  std::vector<std::vector<int64_t>> input_shapes;
  float threshold;
};
