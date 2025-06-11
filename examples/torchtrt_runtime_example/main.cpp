#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include "torch/torch.h"
#include "torch/csrc/inductor/aoti_package/model_package_loader.h"
#include "torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h"

/*
 * This example demonstrates how to load and run a pre-built Torch-TensorRT
 * AOTInductor (AOTI) model package using the PyTorch C++ API.
 *
 * Usage:
 *   torchtrt_runtime_example <path-to-pre-built-trt-aoti module>
 *
 * Steps:
 *   1. Parse the path to the AOTI model package from the command line.
 *   2. Load the model package using AOTIModelPackageLoader.
 *   3. Prepare a random CUDA tensor as input.
 *   4. Run inference using the loaded model.
 *   5. Print the output tensor(s) or an error message if inference fails.
 */

int main(int argc, const char* argv[]) {
  // Check for correct number of command-line arguments
  if (argc < 2) {
    std::cerr << "usage: torchtrt_runtime_example <path-to-pre-built-trt-aoti module>\n";
    return -1;
  }

  // Get the path to the TRT AOTI model package from the command line
  std::string trt_aoti_module_path = argv[1];

  // Enable inference mode for thread-local optimizations
  c10::InferenceMode mode;
  try {
      // Load the AOTI model package
      torch::inductor::AOTIModelPackageLoader runner(trt_aoti_module_path);

      // Create a random input tensor on CUDA with shape [1, 3, 5, 5] and type float32
      std::vector<at::Tensor> inputs = {at::randn({1, 3, 5, 5}, {at::kCUDA}).to(torch::kFloat32)};

      // Run inference using the loaded model
      std::vector<at::Tensor> outputs = runner.run(inputs);

      // Process and print the output tensor(s)
      if (!outputs.empty()) {
          std::cout << "Model output: " << outputs[0] << std::endl;
      } else {
          std::cerr << "No output tensors received!" << std::endl;
      }

  } catch (const c10::Error& e) {
      // Handle errors from the PyTorch C++ API
      std::cerr << "Error running model: " << e.what() << std::endl;
      return 1;
  } catch (const std::exception& e) {
      // Handle other standard exceptions
      std::cerr << "An unexpected error occurred: " << e.what() << std::endl;
      return 1;
  }

  return 0;
}
