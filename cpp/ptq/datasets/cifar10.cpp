#include "cpp/ptq/datasets/cifar10.h"

#include "torch/data/example.h"
#include "torch/types.h"

#include <iostream>
#include <cstddef>
#include <fstream>
#include <string>
#include <vector>
#include <utility>
#include <sstream>
#include <memory>

namespace datasets {
namespace {
constexpr const char* kTrainFilenamePrefix = "data_batch_";
constexpr const uint32_t kNumTrainFiles = 5;
constexpr const char* kTestFilename = "test_batch.bin";
constexpr const size_t kLabelSize = 1; // B
constexpr const size_t kImageSize = 3072; // B
constexpr const size_t kImageDim = 32;
constexpr const size_t kImageChannels = 3;
constexpr const size_t kBatchSize = 10000;

std::pair<torch::Tensor, torch::Tensor> read_batch(const std::string& path) {
    std::ifstream batch;
    batch.open(path, std::ios::in|std::ios::binary|std::ios::ate);

    auto file_size = batch.tellg();
    std::unique_ptr<char[]> buf(new char[file_size]);

    batch.seekg(0, std::ios::beg);
    batch.read(buf.get(), file_size);
    batch.close();

    std::vector<uint8_t> labels;
    std::vector<torch::Tensor> images;
    labels.reserve(kBatchSize);
    images.reserve(kBatchSize);

    for (size_t i = 0; i < kBatchSize; i++) {
        uint8_t label = buf[i * (kImageSize + kLabelSize)];
        std::vector<uint8_t> image;
        image.reserve(kImageSize);
        std::copy(&buf[i * (kImageSize + kLabelSize) + 1], &buf[i * (kImageSize + kLabelSize) + kImageSize], std::back_inserter(image));
        labels.push_back(label);
        auto image_tensor = torch::from_blob(image.data(),
                                             {kImageChannels, kImageDim, kImageDim},
                                             torch::TensorOptions().dtype(torch::kU8)).to(torch::kF32);
        images.push_back(image_tensor);
    }

    auto labels_tensor = torch::from_blob(labels.data(),
                                          {kBatchSize},
                                          torch::TensorOptions().dtype(torch::kU8)).to(torch::kF32);
    assert(labels_tensor.size(0) == kBatchSize);

    auto images_tensor = torch::stack(images);
    assert(images_tensor.size(0) == kBatchSize);

    return std::make_pair(images_tensor, labels_tensor);
}

std::pair<torch::Tensor, torch::Tensor> read_train_data(const std::string& root) {
    torch::Tensor images, targets;
    for(uint32_t i = 1; i <= 5; i++) {
        std::stringstream ss;
        ss << root << '/' << kTrainFilenamePrefix << i << ".bin";
        auto batch = read_batch(ss.str());
        images = torch::stack({images, batch.first});
        targets = torch::stack({targets, batch.second});
    }
    return std::make_pair(images, targets);
}

std::pair<torch::Tensor, torch::Tensor> read_test_data(const std::string& root) {
    std::stringstream ss;
    ss << root << '/' << kTestFilename;
    return read_batch(ss.str());
}
}

CIFAR10::CIFAR10(const std::string& root, Mode mode)
    : mode_(mode) {

    std::pair<torch::Tensor, torch::Tensor> data;
    if (mode_ == Mode::kTrain) {
        data = read_train_data(root);
    } else {
        data = read_test_data(root);
    }

    images_ = std::move(data.first);
    targets_ = std::move(data.second);
}

torch::data::Example<> CIFAR10::get(size_t index) {
    return {images_[index], targets_[index]};
}

c10::optional<size_t> CIFAR10::size() const {
    return images_.size(0);
}

bool CIFAR10::is_train() const noexcept {
    return mode_ == Mode::kTrain;
}

const torch::Tensor& CIFAR10::images() const {
    return images_;
}

const torch::Tensor& CIFAR10::targets() const {
    return targets_;
}

} // namespace datasets

