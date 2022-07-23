#include "ATen/Context.h"
#include "c10/cuda/CUDACachingAllocator.h"
#include "cuda_runtime_api.h"
#include "torch/script.h"

#include "timer.h"
#include "torch_tensorrt/torch_tensorrt.h"

#include <iostream>
#include <memory>
#include <sstream>

#define NUM_WARMUP_RUNS 20
#define NUM_RUNS 100

bool checkRtol(const at::Tensor& diff, const std::vector<at::Tensor> inputs) {
  double maxValue = 0.0;
  for (auto& tensor : inputs) {
    maxValue = fmax(tensor.abs().max().item<float>(), maxValue);
  }
  std::cout << "Max Difference: " << diff.abs().max().item<float>() << std::endl;
  return diff.abs().max().item<float>() <= 2e-6 * maxValue;
}

bool almostEqual(const at::Tensor& a, const at::Tensor& b) {
  return checkRtol(a - b, {a, b});
}

void print_avg_std_dev(std::string type, std::vector<float>& runtimes, uint64_t batch_size) {
  float avg_runtime = std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / runtimes.size();
  float fps = (1000.f / avg_runtime) * batch_size;
  std::cout << "[" << type << "]: batch_size: " << batch_size << "\n    Average latency: " << avg_runtime
            << " ms\n    Average FPS: " << fps << " fps" << std::endl;

  std::vector<float> rt_diff(runtimes.size());
  std::transform(runtimes.begin(), runtimes.end(), rt_diff.begin(), [avg_runtime](float x) { return x - avg_runtime; });
  float rt_sq_sum = std::inner_product(rt_diff.begin(), rt_diff.end(), rt_diff.begin(), 0.0);
  float rt_std_dev = std::sqrt(rt_sq_sum / runtimes.size());

  std::vector<float> fps_diff(runtimes.size());
  std::transform(runtimes.begin(), runtimes.end(), fps_diff.begin(), [fps, batch_size](float x) {
    return ((1000.f / x) * batch_size) - fps;
  });
  float fps_sq_sum = std::inner_product(fps_diff.begin(), fps_diff.end(), fps_diff.begin(), 0.0);
  float fps_std_dev = std::sqrt(fps_sq_sum / runtimes.size());
  std::cout << "    Latency Standard Deviation: " << rt_std_dev << "\n    FPS Standard Deviation: " << fps_std_dev
            << "\n(excluding initial warmup runs)" << std::endl;
}

std::vector<float> benchmark_module(torch::jit::script::Module& mod, std::vector<int64_t> shape) {
  auto execution_timer = timers::PreciseCPUTimer();
  std::vector<float> execution_runtimes;

  for (uint64_t i = 0; i < NUM_WARMUP_RUNS; i++) {
    std::vector<torch::jit::IValue> inputs_ivalues;
    auto in = at::rand(shape, {at::kCUDA});
#ifdef HALF
    in = in.to(torch::kHalf);
#endif
    inputs_ivalues.push_back(in.clone());

    cudaDeviceSynchronize();
    mod.forward(inputs_ivalues);
    cudaDeviceSynchronize();
  }

  for (uint64_t i = 0; i < NUM_RUNS; i++) {
    std::vector<torch::jit::IValue> inputs_ivalues;
    auto in = at::rand(shape, {at::kCUDA});
#ifdef HALF
    in = in.to(torch::kHalf);
#endif
    inputs_ivalues.push_back(in.clone());
    cudaDeviceSynchronize();

    execution_timer.start();
    mod.forward(inputs_ivalues);
    cudaDeviceSynchronize();
    execution_timer.stop();

    auto time = execution_timer.milliseconds();
    execution_timer.reset();
    execution_runtimes.push_back(time);

    c10::cuda::CUDACachingAllocator::emptyCache();
  }
  return execution_runtimes;
}

int main(int argc, const char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage: benchmark <path-to-exported-script-module> <input-size>\n" << std::endl;
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

  std::vector<std::vector<int64_t>> dims;
  for (int i = 2; i < argc; i++) {
    auto arg = std::string(argv[i]);
    arg = arg.substr(1, arg.size() - 1);
    std::istringstream iss(arg);
    std::vector<int64_t> v;
    int64_t n;
    while (iss >> n) {
      v.push_back(n);
    }
    dims.push_back(v);
  }

  at::globalContext().setBenchmarkCuDNN(true);

#ifdef TRT
  auto compile_spec = torch_tensorrt::ts::CompileSpec(dims);

#ifdef HALF
  compile_spec.enabled_precisions.insert(torch::kF16);
#endif

  auto trt_mod = torch_tensorrt::ts::compile(mod, compile_spec);

#ifdef SAVE_ENGINE
  std::cout << "Compiling graph to save as TRT engine (/tmp/engine_converted_from_jit.trt)" << std::endl;
  auto engine = torch_tensorrt::ts::convert_method_to_trt_engine(mod, "forward", compile_spec);
  std::ofstream out("/tmp/engine_converted_from_jit.trt");
  out << engine;
  out.close();
#endif

  auto trt_runtimes = benchmark_module(trt_mod, dims[0]);
  print_avg_std_dev("JIT/TRT", trt_runtimes, dims[0][0]);
#endif

#ifdef HALF
  mod.to(torch::kHalf);
  for (auto layer : mod.named_modules()) {
    if (layer.name.find(".bn") != std::string::npos) {
      layer.value.to(torch::kFloat);
    }
  }
#endif

#ifdef JIT
  auto jit_runtimes = benchmark_module(mod, dims[0]);
  print_avg_std_dev("JIT", jit_runtimes, dims[0][0]);
#endif

  std::cout << "ok\n";
}
