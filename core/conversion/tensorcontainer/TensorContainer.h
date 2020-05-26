#pragma once

#include "NvInfer.h"
#include "torch/custom_class.h"

namespace trtorch {
namespace core {
namespace conversion {

struct TensorContainer : torch::CustomClassHolder {
  int64_t tensor_;
  TensorContainer(int64_t init) : tensor_(init) {}

  c10::intrusive_ptr<TensorContainer> clone() const {
    return c10::make_intrusive<TensorContainer>(tensor_);
  }

  nvinfer1::ITensor* tensor() {
    return reinterpret_cast<nvinfer1::ITensor*>(tensor_);
  }
};

} // conversion
} // core
} // trtorch