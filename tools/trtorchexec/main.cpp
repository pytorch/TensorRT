#include "torch/script.h"

#include "trtorch/trtorch.h"

#include <iostream>
#include <memory>
#include <sstream>

bool checkRtol(const at::Tensor& diff, const std::vector<at::Tensor> inputs) {
  double maxValue = 0.0;
  for (auto& tensor : inputs) {
    maxValue = fmax(tensor.abs().max().item<float>(), maxValue);
  }
  std::cout << "Max Difference: " << diff.abs().max().item<float>() << std::endl;
  return diff.abs().max().item<float>() <= 2e-5 * maxValue;
}

bool almostEqual(const at::Tensor& a, const at::Tensor& b) {
  return checkRtol(a - b, {a, b});
}

int main(int argc, const char* argv[]) {
  if (argc < 3) {
    std::cerr
        << "usage: trtorchexec <path-to-exported-script-module> <input-size>\n"
        << "       trtorchexec <path-to-exported-script-module> <min-input-size> <opt-input-size> <max-input-size>\n";
    return -1;
  }

  torch::jit::Module mod;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    mod = torch::jit::load(argv[1]);
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  mod.to(at::kCUDA);
  mod.eval();

  std::vector<std::vector<int64_t>> dims;
  for (int i = 2; i < argc; i++) {
    auto arg = std::string(argv[i]);
    arg = arg.substr(1, arg.size() - 1);
    std::istringstream iss(arg);
    std::vector<int64_t> v;
    std::cout << '[';
    int64_t n;
    while (iss >> n) {
      v.push_back(n);
      std::cout << n << ',';
    }
    std::cout << ']' << std::endl;
    dims.push_back(v);
  }

  auto compile_spec = trtorch::CompileSpec(dims);

  std::cout << "Checking operator support" << std::endl;
  if (!trtorch::CheckMethodOperatorSupport(mod, "forward")) {
    std::cerr << "Method is not currently supported by TRTorch" << std::endl;
    return -1;
  }

  std::cout << "Compiling graph to save as TRT engine (/tmp/engine_converted_from_jit.trt)" << std::endl;
  auto engine = trtorch::ConvertGraphToTRTEngine(mod, "forward", compile_spec);
  std::ofstream out("/tmp/engine_converted_from_jit.trt");
  out << engine;
  out.close();

  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  auto in = at::randint(5, dims[0], {at::kCUDA});
  jit_inputs_ivalues.push_back(in.clone());
  trt_inputs_ivalues.push_back(in.clone());

  torch::jit::IValue jit_results_ivalues = mod.forward(jit_inputs_ivalues);
  std::vector<at::Tensor> jit_results;
  if (jit_results_ivalues.isTensor()) {
    jit_results.push_back(jit_results_ivalues.toTensor());
  } else {
    auto results = jit_results_ivalues.toTuple()->elements();
    for (auto r : results) {
      jit_results.push_back(r.toTensor());
    }
  }

  std::cout << "Compiling graph as module" << std::endl;
  auto trt_mod = trtorch::CompileGraph(mod, compile_spec);
  std::cout << "Running TRT module" << std::endl;
  torch::jit::IValue trt_results_ivalues = trt_mod.forward(trt_inputs_ivalues);
  std::vector<at::Tensor> trt_results;
  if (trt_results_ivalues.isTensor()) {
    trt_results.push_back(trt_results_ivalues.toTensor());
  } else {
    auto results = trt_results_ivalues.toTuple()->elements();
    for (auto r : results) {
      trt_results.push_back(r.toTensor());
    }
  }

  for (size_t i = 0; i < trt_results.size(); i++) {
    almostEqual(jit_results[i], trt_results[i].reshape_as(jit_results[i]));
  }

  std::cout << "Converted Engine saved to /tmp/engine_converted_from_jit.trt" << std::endl;

  trt_mod.save("/tmp/ts_trt.ts");
  std::cout << "Compiled TorchScript program saved to /tmp/ts_trt.ts" << std::endl;
  std::cout << "ok\n";
}
