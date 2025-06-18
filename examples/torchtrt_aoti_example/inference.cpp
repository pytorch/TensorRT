#include <iostream>
#include <vector>

#include "torch/torch.h"
#include "torch/csrc/inductor/aoti_package/model_package_loader.h"

int main(int argc, const char* argv[]) {
  // Check for correct number of command-line arguments
  std::string trt_aoti_module_path = "model.pt2";

  if (argc == 2) {
    trt_aoti_module_path = argv[1];
  }

  std::cout << trt_aoti_module_path << std::endl;

  // Get the path to the TRT AOTI model package from the command line
  c10::InferenceMode mode;

  torch::inductor::AOTIModelPackageLoader loader(trt_aoti_module_path);
  // Assume running on CUDA
  std::vector<torch::Tensor> inputs = {torch::randn({8, 10}, at::kCUDA)};
  std::vector<torch::Tensor> outputs = loader.run(inputs);
  std::cout << "Result from the first inference:"<< std::endl;
  std::cout << outputs << std::endl;

  // The second inference uses a different batch size and it works because we
  // specified that dimension as dynamic when compiling model.pt2.
  std::cout << "Result from the second inference:"<< std::endl;
  // Assume running on CUDA
  std::cout << loader.run({torch::randn({1, 10}, at::kCUDA)}) << std::endl;

  return 0;
}
