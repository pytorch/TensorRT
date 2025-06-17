#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include "torch/script.h"

int main(int argc, const char* argv[]) {
  if (argc < 2) {
    std::cerr << "usage: samplertapp <path-to-pre-built-trt-ts module>\n";
    return -1;
  }

  std::string trt_ts_module_path = argv[1];

  torch::jit::Module trt_ts_mod;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    trt_ts_mod = torch::jit::load(trt_ts_module_path);
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model from : " << trt_ts_module_path << std::endl;
    return -1;
  }

  std::cout << "Running TRT engine" << std::endl;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  trt_inputs_ivalues.push_back(at::randint(-5, 5, {1, 3, 5, 5}, {at::kCUDA}).to(torch::kFloat32));
  torch::jit::IValue trt_results_ivalues = trt_ts_mod.forward(trt_inputs_ivalues);
  std::cout << "==================TRT outputs================" << std::endl;
  std::cout << trt_results_ivalues << std::endl;
  std::cout << "=============================================" << std::endl;
  std::cout << "TRT engine execution completed. " << std::endl;
}
