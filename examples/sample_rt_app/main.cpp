#include <iostream>
#include <fstream>
#include <memory>
#include <sstream>
#include <vector>
#include "torch/script.h"

// Load the TRT engine from engine_path
std::vector<char> loadEngine(std::string engine_path){
  std::ifstream engineFile(engine_path, std::ios::binary);
  if (!engineFile)
  {
      std::cerr << "Error opening TensorRT Engine file at : " << engine_path << std::endl;
  }

  engineFile.seekg(0, engineFile.end);
  long int fsize = engineFile.tellg();
  engineFile.seekg(0, engineFile.beg);

  std::vector<char> engineData(fsize);
  engineFile.read(engineData.data(), fsize);
  if (!engineFile)
  {
      std::cerr << "Error loading engine from: " << engine_path << std::endl;
  }

  return engineData;
}

int main(int argc, const char* argv[]) {
  if (argc < 2) {
    std::cerr
        << "usage: samplertapp <path-to-pre-built-trt-ts module>\n";
    return -1;
  }

  std::string trt_ts_module_path = argv[1];
  // auto engineData = loadEngine(engine_path);

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
