// #include "examples/sample_rt_app/trtorch/include/trtorch/core/runtime/runtime.h"
#include "trtorch/core/runtime/runtime.h"
#include <iostream>
#include <fstream>
#include <memory>
#include <sstream>
#include <vector>
#include "torch/script.h"
#include "trtorch/include/trtorch/trtorch.h"

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
        << "usage: samplertapp <path-to-pre-built-trt-engine>\n";
    return -1;
  }

  std::string engine_path = argv[1];
  auto engineData = loadEngine(engine_path);

  std::cout << "Running TRT engine" << std::endl;
  auto engine_ptr = c10::make_intrusive<TRTEngine>("test_engine", engineData.data());
  auto inputs = at::randint(-5, 5, {1, 3, 5, 5}, {at::kCUDA});
  auto outputs = trtorch::core::runtime::execute_engine(inputs, engine_ptr);
  std::cout << "TRT engine execution completed. " << std::endl;
}
