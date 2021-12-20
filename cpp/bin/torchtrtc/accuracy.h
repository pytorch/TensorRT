#pragma once

#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <vector>

#include "torch/script.h"
#include "torch/torch.h"

namespace torchtrtc {
namespace accuracy {

bool check_rtol(const at::Tensor& diff, const std::vector<at::Tensor> inputs, float threshold);
bool almost_equal(const at::Tensor& a, const at::Tensor& b, float threshold);

} // namespace accuracy
} // namespace torchtrtc