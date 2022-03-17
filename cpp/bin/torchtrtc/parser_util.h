#pragma once
#include <stdlib.h>
#include <iostream>
#include <sstream>

#include "NvInfer.h"
#include "third_party/args/args.hpp"
#include "torch/script.h"
#include "torch/torch.h"

#include "torch_tensorrt/logging.h"
#include "torch_tensorrt/ptq.h"
#include "torch_tensorrt/torch_tensorrt.h"

namespace torchtrtc {
namespace parserutil {

// String to TensorFormat Enum
torchtrt::TensorFormat parse_tensor_format(std::string str);

// String to data type
torchtrt::DataType parse_dtype(std::string dtype_str);

// String to a vector of ints which represents a dimension spec
std::vector<int64_t> parse_single_dim(std::string shape_str);

// String to a vector of 3 dimension specs specs (each a vector of ints)
std::vector<std::vector<int64_t>> parse_dynamic_dim(std::string shape_str);

// String to a torchtrt::Input
torchtrt::Input parse_input(std::string input_specs);

} // namespace parserutil
} // namespace torchtrtc