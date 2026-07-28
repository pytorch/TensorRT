#include "accuracy_test.h"
#include "datasets/cifar10.h"
#include "torch/torch.h"

TEST_P(AccuracyTests, DLAFP16AccuracyIsClose) {
  auto eval_dataset =
      datasets::CIFAR10("tests/accuracy/datasets/data/cifar-10-batches-bin/", datasets::CIFAR10::Mode::kTest)
          .use_subset(3200)
          .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
          .map(torch::data::transforms::Stack<>());
  auto eval_dataloader = torch::data::make_data_loader(
      std::move(eval_dataset), torch::data::DataLoaderOptions().batch_size(32).workers(2));

  // Check the FP32 accuracy in JIT
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

  std::vector<std::vector<int64_t>> input_shape = {{32, 3, 32, 32}};
  auto compile_spec = torch_tensorrt::ts::CompileSpec({input_shape});
  compile_spec.enabled_precisions = {torch::kF16};
  compile_spec.device.device_type = torch_tensorrt::CompileSpec::Device::DeviceType::kDLA;
  compile_spec.device.gpu_id = 0;
  compile_spec.device.dla_core = 1;
  compile_spec.device.allow_gpu_fallback = true;

  auto trt_mod = torch_tensorrt::ts::compile(mod, compile_spec);

  torch::Tensor trt_correct = torch::zeros({1}, {torch::kCUDA}), trt_total = torch::zeros({1}, {torch::kCUDA});
  for (auto batch : *eval_dataloader) {
    auto images = batch.data.to(torch::kCUDA).to(torch::kF16);
    auto targets = batch.target.to(torch::kCUDA).to(torch::kF16);

    auto outputs = trt_mod.forward({images});
    auto predictions = std::get<1>(torch::max(outputs.toTensor(), 1, false));
    predictions = predictions.reshape(predictions.sizes()[0]);

    trt_total += targets.sizes()[0];
    trt_correct += torch::sum(torch::eq(predictions, targets));
  }

  torch::Tensor trt_accuracy = (trt_correct / trt_total) * 100;

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_accuracy, trt_accuracy, 3));
}

INSTANTIATE_TEST_SUITE_P(
    DLAFP16AccuracyIsCloseSuite,
    AccuracyTests,
    testing::Values("tests/accuracy/vgg16_cifar10.jit.pt"));
