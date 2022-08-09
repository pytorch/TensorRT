#include "torch/script.h"
#include "torch/torch.h"
#include "torch_tensorrt/torch_tensorrt.h"

#include "examples/int8/benchmark/benchmark.h"
#include "examples/int8/datasets/cifar10.h"

#include <sys/stat.h>
#include <iostream>
#include <memory>
#include <sstream>

namespace F = torch::nn::functional;

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

torch::jit::Module compile_int8_qat_model(const std::string& data_dir, torch::jit::Module& mod) {
  std::vector<torch_tensorrt::Input> inputs = {
      torch_tensorrt::Input(std::vector<int64_t>({32, 3, 32, 32}), torch_tensorrt::DataType::kFloat)};
  /// Configure settings for compilation
  auto compile_spec = torch_tensorrt::ts::CompileSpec(inputs);
  /// Set operating precision to INT8
  compile_spec.enabled_precisions.insert(torch::kI8);

#ifdef SAVE_ENGINE
  std::cout << "Compiling graph to save as TRT engine (/tmp/engine_converted_from_jit.trt)" << std::endl;
  auto engine = torch_tensorrt::ts::convert_method_to_trt_engine(mod, "forward", compile_spec);
  std::ofstream out("/tmp/int8_engine_converted_from_jit.trt");
  out << engine;
  out.close();
#endif

  std::cout << "Compiling and quantizing module" << std::endl;
  auto trt_mod = torch_tensorrt::ts::compile(mod, compile_spec);
  return std::move(trt_mod);
}

int main(int argc, const char* argv[]) {
  at::globalContext().setBenchmarkCuDNN(true);

  if (argc < 3) {
    std::cerr << "usage: qat <path-to-module> <path-to-cifar10>\n";
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

  mod.eval();

  /// Create the calibration dataset
  const std::string data_dir = std::string(argv[2]);

  /// Dataloader moved into calibrator so need another for inference
  auto eval_dataset = datasets::CIFAR10(data_dir, datasets::CIFAR10::Mode::kTest)
                          .use_subset(3200)
                          .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
                          .map(torch::data::transforms::Stack<>());
  auto eval_dataloader = torch::data::make_data_loader(
      std::move(eval_dataset), torch::data::DataLoaderOptions().batch_size(32).workers(2));

  /// Check the FP32 accuracy in JIT
  torch::Tensor jit_correct = torch::zeros({1}, {torch::kCUDA}), jit_total = torch::zeros({1}, {torch::kCUDA});
  for (auto batch : *eval_dataloader) {
    auto images = batch.data.to(torch::kCUDA);
    auto targets = batch.target.to(torch::kCUDA);

    auto outputs = mod.forward({images});
    auto predictions = std::get<1>(torch::max(outputs.toTensor(), 1, false));

    jit_total += targets.sizes()[0];
    jit_correct += torch::sum(torch::eq(predictions, targets));
  }
  torch::Tensor jit_accuracy = (jit_correct / jit_total) * 100;

  /// Compile Graph
  auto trt_mod = compile_int8_qat_model(data_dir, mod);

  /// Check the INT8 accuracy in TRT
  torch::Tensor trt_correct = torch::zeros({1}, {torch::kCUDA}), trt_total = torch::zeros({1}, {torch::kCUDA});
  for (auto batch : *eval_dataloader) {
    auto images = batch.data.to(torch::kCUDA);
    auto targets = batch.target.to(torch::kCUDA);

    auto outputs = trt_mod.forward({images});
    auto predictions = std::get<1>(torch::max(outputs.toTensor(), 1, false));
    predictions = predictions.reshape(predictions.sizes()[0]);

    trt_total += targets.sizes()[0];
    trt_correct += torch::sum(torch::eq(predictions, targets)).item().toFloat();
  }
  torch::Tensor trt_accuracy = (trt_correct / trt_total) * 100;

  std::cout << "Accuracy of JIT model on test set: " << jit_accuracy.item().toFloat() << "%" << std::endl;
  std::cout << "Accuracy of quantized model on test set: " << trt_accuracy.item().toFloat() << "%" << std::endl;

  /// Time execution in JIT-FP32 and TRT-INT8
  std::vector<std::vector<int64_t>> dims = {{32, 3, 32, 32}};

  auto jit_runtimes = benchmark_module(mod, dims[0]);
  print_avg_std_dev("JIT model FP32", jit_runtimes, dims[0][0]);

  auto trt_runtimes = benchmark_module(trt_mod, dims[0]);
  print_avg_std_dev("TRT quantized model", trt_runtimes, dims[0][0]);
  trt_mod.save("/tmp/qat_vgg16.trt.ts");
}
