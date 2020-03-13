#pragma once

#include <vector>
#include <string>

#include "ATen/Tensor.h"
#include "core/util/prelude.h"
#include "core/conversion/conversion.h"

namespace trtorch {
namespace tests {
namespace util {

bool almostEqual(const at::Tensor& a, const at::Tensor& b);

bool exactlyEqual(const at::Tensor& a, const at::Tensor& b);

std::vector<at::Tensor> RunEngine(std::string& eng, std::vector<at::Tensor> inputs);

// Runs an arbitrary JIT graph and returns results  
std::vector<at::Tensor> RunGraph(std::shared_ptr<torch::jit::Graph>& g,
                                 core::conversion::GraphParams& named_params,
                                 std::vector<at::Tensor> inputs);

// Runs an arbitrary JIT graph by converting it to TensorRT and running inference
// and returns results
std::vector<at::Tensor> RunGraphEngine(std::shared_ptr<torch::jit::Graph>& g,
                                       core::conversion::GraphParams& named_params,
                                       std::vector<at::Tensor> inputs);

// Run the forward method of a module and return results
torch::jit::IValue RunModuleForward(torch::jit::script::Module& mod,
                                    std::vector<torch::jit::IValue> inputs);

// Convert the forward module to a TRT engine and return results
std::vector<at::Tensor> RunModuleForwardAsEngine(torch::jit::script::Module& mod,
                                                 std::vector<at::Tensor> inputs);


} // namespace util
} // namespace tests
} // namespace trtorch
