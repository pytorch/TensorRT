#include "torch/script.h"
#include "torch/torch.h"
#include "trtorch/trtorch.h"

#include "NvInfer.h"

#include "datasets/cifar10.h"
#include "timer.h"

#include <iostream>
#include <sstream>
#include <memory>
#include <sys/stat.h>

int main(int argc, const char* argv[]) {
    trtorch::logging::set_reportable_log_level(trtorch::logging::kINFO);
    if (argc < 3) {
        std::cerr << "usage: ptq <path-to-module> <path-to-cifar10>\n";
        return -1;
    }

    torch::jit::script::Module mod;
    try {
         // Deserialize the ScriptModule from a file using torch::jit::load().
         mod = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
         std::cerr << "error loading the model\n";
         return -1;
    }

    // Create the calibration dataset
    const std::string data_dir = std::string(argv[2]);
    auto calibration_dataset = datasets::CIFAR10(data_dir, datasets::CIFAR10::Mode::kTest)
                                    .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465},
                                                                              {0.2023, 0.1994, 0.2010}))
                                    .map(torch::data::transforms::Stack<>());
    auto calibration_dataloader = torch::data::make_data_loader(std::move(calibration_dataset), torch::data::DataLoaderOptions()
                                                                                                    .batch_size(32)
                                                                                                    .workers(2));

    std::string calibration_cache_file = "/tmp/vgg16_TRT_ptq_calibration.cache";

    auto calibrator = trtorch::ptq::make_int8_calibrator(std::move(calibration_dataloader), calibration_cache_file, true);
    //auto calibrator = trtorch::ptq::make_int8_cache_calibrator(calibration_cache_file);


    std::vector<std::vector<int64_t>> input_shape = {{32, 3, 32, 32}};
    // Configure settings for compilation
    auto extra_info = trtorch::ExtraInfo({input_shape});
    // Set operating precision to INT8
    extra_info.op_precision = torch::kChar;
    // Use the TensorRT Entropy Calibrator
    extra_info.ptq_calibrator = calibrator;
    // Increase the default workspace size;
    extra_info.workspace_size = 1 << 30;

    mod.eval();

    // Dataloader moved into calibrator so need another for inference
    auto eval_dataset = datasets::CIFAR10(data_dir, datasets::CIFAR10::Mode::kTest)
                                    .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465},
                                                                              {0.2023, 0.1994, 0.2010}))
                                    .map(torch::data::transforms::Stack<>());
    auto eval_dataloader = torch::data::make_data_loader(std::move(eval_dataset), torch::data::DataLoaderOptions()
                                                                                                    .batch_size(32)
                                                                                                    .workers(2));

    // Check the FP32 accuracy in JIT
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

    // Compile Graph
    auto trt_mod = trtorch::CompileGraph(mod, extra_info);

    // Check the INT8 accuracy in TRT
    correct = 0.0;
    total = 0.0;
    for (auto batch : *eval_dataloader) {
        auto images = batch.data.to(torch::kCUDA);
        auto targets = batch.target.to(torch::kCUDA);

        auto outputs = trt_mod.forward({images});
        auto predictions = std::get<1>(torch::max(outputs.toTensor(), 1, false));

        total += targets.sizes()[0];
        correct += torch::sum(torch::eq(predictions, targets)).item().toFloat();
        std::cout << total << " " << correct << std::endl;
    }
    std::cout << total << " " << correct << std::endl;
    std::cout << "Accuracy of quantized model on test set: " << 100 * (correct / total) << "%" << std::endl;

    // Time execution in INT8
    auto execution_timer = timers::PreciseCPUTimer();
    auto images = (*(*eval_dataloader).begin()).data.to(torch::kCUDA);

    execution_timer.start();
    trt_mod.forward({images});
    execution_timer.stop();

    std::cout << "Latency of quantized model (Batch Size 32): " << execution_timer.milliseconds() << "ms" << std::endl;
}
