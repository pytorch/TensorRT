#include <utility>
#include "c10/cuda/CUDACachingAllocator.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/script.h"
#include "torch_tensorrt/torch_tensorrt.h"

// TODO: Extend this to support other datasets
class AccuracyTests : public testing::TestWithParam<std::string> {
 public:
  void SetUp() override {
    auto params = GetParam();
    auto module_path = params;
    try {
      // Deserialize the ScriptModule from a file using torch::jit::load().
      mod = torch::jit::load(module_path);
    } catch (const c10::Error& e) {
      std::cerr << "error loading the model\n";
      return;
    }
    mod.eval();
  }

  void TearDown() {
    c10::cuda::CUDACachingAllocator::emptyCache();
  }

 protected:
  torch::jit::script::Module mod;
};
