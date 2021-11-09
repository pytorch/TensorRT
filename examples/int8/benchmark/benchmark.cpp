#include "ATen/Context.h"
#include "c10/cuda/CUDACachingAllocator.h"
#include "cuda_runtime_api.h"
#include "torch/script.h"
#include "torch/torch.h"
#include "torch_tensorrt/torch_tensorrt.h"

#include "timer.h"

#define NUM_WARMUP_RUNS 20
#define NUM_RUNS 100

// Benchmaking code
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
