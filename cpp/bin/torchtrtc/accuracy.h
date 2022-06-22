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
bool almost_equal(const at::Tensor& computed_tensor, const at::Tensor& gt_tensor, float atol = 1e-8, float rtol = 1e-5);

} // namespace accuracy
} // namespace torchtrtc
