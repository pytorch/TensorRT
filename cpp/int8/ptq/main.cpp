#include "torch/script.h"
#include "torch/torch.h"
#include "trtorch/ptq.h"
#include "trtorch/trtorch.h"

#include "NvInfer.h"

#include "cpp/int8/benchmark/benchmark.h"
#include "cpp/int8/datasets/cifar10.h"

#include <sys/stat.h>
#include <iostream>
#include <memory>
#include <sstream>

namespace F = torch::nn::functional;

// Actual PTQ application code
struct Resize : public torch::data::transforms::TensorTransform<torch::Tensor> {
  Resize(std::vector<int64_t> new_size) : new_size_(new_size) {}

  torch::Tensor operator()(torch::Tensor input) {
    input = input.unsqueeze(0);
    auto upsampled =
        F::interpolate(input, F::InterpolateFuncOptions().size(new_size_).align_corners(false).mode(torch::kBilinear));
    return upsampled.squeeze(0);
  }

  std::vector<int64_t> new_size_;
};

torch::jit::Module compile_int8_model(const std::string& data_dir, torch::jit::Module& mod) {
  auto calibration_dataset =
      datasets::CIFAR10(data_dir, datasets::CIFAR10::Mode::kTest)
          .use_subset(320)
          .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
          .map(torch::data::transforms::Stack<>());
  auto calibration_dataloader = torch::data::make_data_loader(
      std::move(calibration_dataset), torch::data::DataLoaderOptions().batch_size(32).workers(2));

  std::string calibration_cache_file = "/tmp/vgg16_TRT_ptq_calibration.cache";

  auto calibrator = trtorch::ptq::make_int8_calibrator(std::move(calibration_dataloader), calibration_cache_file, true);

  std::vector<std::vector<int64_t>> input_shape = {{32, 3, 32, 32}};
  /// Configure settings for compilation
  auto compile_spec = trtorch::CompileSpec({input_shape});
  /// Set operating precision to INT8
  compile_spec.enabled_precisions.insert(torch::kI8);
  /// Use the TensorRT Entropy Calibrator
  compile_spec.ptq_calibrator = calibrator;
  /// Set max batch size for the engine
  compile_spec.max_batch_size = 32;
  /// Set a larger workspace
  compile_spec.workspace_size = 1 << 28;

  mod.eval();

#ifdef SAVE_ENGINE
  std::cout << "Compiling graph to save as TRT engine (/tmp/engine_converted_from_jit.trt)" << std::endl;
  auto engine = trtorch::ConvertGraphToTRTEngine(mod, "forward", compile_spec);
  std::ofstream out("/tmp/engine_converted_from_jit.trt");
  out << engine;
  out.close();
#endif

  std::cout << "Compiling and quantizing module" << std::endl;
  auto trt_mod = trtorch::CompileGraph(mod, compile_spec);
  return std::move(trt_mod);
}

int main(int argc, const char* argv[]) {
  at::globalContext().setBenchmarkCuDNN(true);

  if (argc < 3) {
    std::cerr << "usage: ptq <path-to-module> <path-to-cifar10>\n";
    return -1;
  }

  torch::jit::Module mod;
  try {
    /// Deserialize the ScriptModule from a file using torch::jit::load().
    mod = torch::jit::load(argv[1]);
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  /// Create the calibration dataset
  const std::string data_dir = std::string(argv[2]);
  auto trt_mod = compile_int8_model(data_dir, mod);

  /// Dataloader moved into calibrator so need another for inference
  auto eval_dataset = datasets::CIFAR10(data_dir, datasets::CIFAR10::Mode::kTest)
                          .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
                          .map(torch::data::transforms::Stack<>());
  auto eval_dataloader = torch::data::make_data_loader(
      std::move(eval_dataset), torch::data::DataLoaderOptions().batch_size(32).workers(2));

  /// Check the FP32 accuracy in JIT
  float correct = 0.0, total = 0.0;
  for (auto batch : *eval_dataloader) {
    auto images = batch.data.to(torch::kCUDA);
    auto targets = batch.target.to(torch::kCUDA);

    auto outputs = mod.forward({images});
    auto predictions = std::get<1>(torch::max(outputs.toTensor(), 1, false));

    total += targets.sizes()[0];
    correct += torch::sum(torch::eq(predictions, targets)).item().toFloat();
  }
  std::cout << "Accuracy of JIT model on test set: " << 100 * (correct / total) << "%" << std::endl;

  /// Check the INT8 accuracy in TRT
  correct = 0.0;
  total = 0.0;
  for (auto batch : *eval_dataloader) {
    auto images = batch.data.to(torch::kCUDA);
    auto targets = batch.target.to(torch::kCUDA);

    if (images.sizes()[0] < 32) {
      /// To handle smaller batches util Optimization profiles work with Int8
      auto diff = 32 - images.sizes()[0];
      auto img_padding = torch::zeros({diff, 3, 32, 32}, {torch::kCUDA});
      auto target_padding = torch::zeros({diff}, {torch::kCUDA});
      images = torch::cat({images, img_padding}, 0);
      targets = torch::cat({targets, target_padding}, 0);
    }

    auto outputs = trt_mod.forward({images});
    auto predictions = std::get<1>(torch::max(outputs.toTensor(), 1, false));
    predictions = predictions.reshape(predictions.sizes()[0]);

    if (predictions.sizes()[0] != targets.sizes()[0]) {
      /// To handle smaller batches util Optimization profiles work with Int8
      predictions = predictions.slice(0, 0, targets.sizes()[0]);
    }

    total += targets.sizes()[0];
    correct += torch::sum(torch::eq(predictions, targets)).item().toFloat();
  }
  std::cout << "Accuracy of quantized model on test set: " << 100 * (correct / total) << "%" << std::endl;

  /// Time execution in JIT-FP32 and TRT-INT8
  std::vector<std::vector<int64_t>> dims = {{32, 3, 32, 32}};

  auto jit_runtimes = benchmark_module(mod, dims[0]);
  print_avg_std_dev("JIT model FP32", jit_runtimes, dims[0][0]);

  auto trt_runtimes = benchmark_module(trt_mod, dims[0]);
  print_avg_std_dev("TRT quantized model", trt_runtimes, dims[0][0]);
}
