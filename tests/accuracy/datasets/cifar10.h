#pragma once

#include "torch/data/datasets/base.h"
#include "torch/data/example.h"
#include "torch/types.h"

#include <cstddef>
#include <string>

namespace datasets {
// The CIFAR10 Dataset
class CIFAR10 : public torch::data::datasets::Dataset<CIFAR10> {
 public:
  // The mode in which the dataset is loaded
  enum class Mode { kTrain, kTest };

  // Loads CIFAR10 from un-tarred file
  // Dataset can be found
  // https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz Root path should be
  // the directory that contains the content of tarball
  explicit CIFAR10(const std::string& root, Mode mode = Mode::kTrain);

  // Returns the pair at index in the dataset
  torch::data::Example<> get(size_t index) override;

  // The size of the dataset
  c10::optional<size_t> size() const override;

  // The mode the dataset is in
  bool is_train() const noexcept;

  // Returns all images stacked into a single tensor
  const torch::Tensor& images() const;

  // Returns all targets stacked into a single tensor
  const torch::Tensor& targets() const;

  // Trims the dataset to the first n pairs
  CIFAR10&& use_subset(int64_t new_size);

 private:
  Mode mode_;
  torch::Tensor images_, targets_;
};
} // namespace datasets
