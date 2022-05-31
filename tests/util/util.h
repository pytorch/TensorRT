#pragma once

#include <ATen/ATen.h>
#include <string>
#include <vector>
#include "ATen/Tensor.h"
#include "core/ir/ir.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace tests {
namespace util {

bool almostEqual(const at::Tensor& computed_tensor, const at::Tensor& gt_tensor, float atol = 1e-8, float rtol = 1e-5);

bool exactlyEqual(const at::Tensor& a, const at::Tensor& b);

std::vector<at::Tensor> RunEngine(std::string& eng, std::vector<at::Tensor> inputs);

// Runs an arbitrary JIT graph and returns results
std::vector<at::Tensor> RunGraph(
    std::shared_ptr<torch::jit::Graph>& g,
    core::ir::StaticParams& named_params,
    std::vector<at::Tensor> inputs);

// Runs an arbitrary JIT graph by converting it to TensorRT and running
// inference and returns results
std::vector<at::Tensor> RunGraphEngine(
    std::shared_ptr<torch::jit::Graph>& g,
    core::ir::StaticParams& named_params,
    std::vector<at::Tensor> inputs,
    nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT);

// Runs an arbitrary JIT graph with dynamic input sizes by converting it to
// TensorRT and running inference and returns results
std::vector<at::Tensor> RunGraphEngineDynamic(
    std::shared_ptr<torch::jit::Graph>& g,
    core::ir::StaticParams& named_params,
    std::vector<at::Tensor> inputs,
    bool dynamic_batch = false);

// Run the forward method of a module and return results
torch::jit::IValue RunModuleForward(torch::jit::Module& mod, std::vector<torch::jit::IValue> inputs);

// Convert the forward module to a TRT engine and return results
std::vector<at::Tensor> RunModuleForwardAsEngine(torch::jit::Module& mod, std::vector<at::Tensor> inputs);

// Runs evaluatable graphs through the compiler evaluator library and returns results
std::vector<torch::jit::IValue> EvaluateGraph(const torch::jit::Block* b, std::vector<torch::jit::IValue> inputs);

// Runs evaluatable graphs through the JIT interpreter and returns results
std::vector<torch::jit::IValue> EvaluateGraphJIT(
    std::shared_ptr<torch::jit::Graph>& g,
    std::vector<torch::jit::IValue> inputs);
} // namespace util
} // namespace tests
} // namespace torch_tensorrt
