#include "torch/script.h"
#include "torch/csrc/api/include/torch/data/datasets/mnist.h"
#include "trtorch/trtorch.h"

#include <iostream>
#include <sstream>
#include <memory>

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        std::cerr << "usage: ptq <path-to-module> <path-to-mnist>\n";
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

    const std::string data_dir = std::string(argv[2]);
    auto calibration_dataset = torch::data::datasets::MNIST(data_dir, torch::data::datasets::MNIST::Mode::kTest)
                                    .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                                    .map(torch::data::transforms::Stack<>());
    auto calibration_dataloader = torch::data::make_data_loader(std::move(calibration_dataset), torch::data::DataLoaderOptions()
                                                                                                    .batch_size(32)
                                                                                                    .workers(1))

    for (auto batch : batched_calibration_dataset) {
        std::cout << batch.data().sizes() << std::endl;
    }
}
