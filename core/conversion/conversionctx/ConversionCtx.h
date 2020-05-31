#pragma once

#include <map>
#include <unordered_map>
#include <memory>

#include "torch/csrc/jit/ir/ir.h"
#include "NvInfer.h"

#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace conversion {

struct BuilderSettings {
    // Defaults should reflect TensorRT defaults for BuilderConfig
    nvinfer1::DataType op_precision = nvinfer1::DataType::kFLOAT;
    bool refit = false;
    bool debug = false;
    bool strict_types = false;
    bool allow_gpu_fallback = true;
    nvinfer1::DeviceType device = nvinfer1::DeviceType::kGPU;
    nvinfer1::EngineCapability capability = nvinfer1::EngineCapability::kDEFAULT;
    nvinfer1::IInt8Calibrator* calibrator = nullptr;
    uint64_t num_min_timing_iters = 2;
    uint64_t num_avg_timing_iters = 1;
    uint64_t workspace_size = 0;
    uint64_t max_batch_size = 0;

    BuilderSettings() = default;
    BuilderSettings(const BuilderSettings& other) = default;
    friend std::ostream& operator<<(std::ostream& os, const BuilderSettings& s);
};

struct ConversionCtx {
    ConversionCtx(BuilderSettings settings);
    std::string SerializeEngine();
    nvinfer1::ITensor* AssociateValueAndTensor(const torch::jit::Value* value, nvinfer1::ITensor* tensor);
    torch::jit::IValue* AssociateValueAndIValue(const torch::jit::Value* value, torch::jit::IValue tensor);
    bool CheckLayerAddition(const torch::jit::Node* n);

    ~ConversionCtx();

    bool input_is_dynamic = false;
    nvinfer1::IBuilder* builder;
    nvinfer1::INetworkDefinition* net;
    nvinfer1::IBuilderConfig* cfg;
    nvinfer1::DataType input_type;
    nvinfer1::DataType op_precision;
    BuilderSettings settings;
    util::logging::TRTorchLogger logger;
    // Pointers to data that needs to remain alive until conversion is done
    // All data will be freed when the destructor is called
    // The weights class is the main consumer of this, each time a weight object
    // is constructed from a PyTorch Tensor it allocates the data here to store a
    // copy of the values
    std::vector<void*> builder_resources;

    std::unordered_map<const torch::jit::Value*, nvinfer1::ITensor*> value_tensor_map;
    std::unordered_map<const torch::jit::Value*, torch::jit::IValue> evaluated_value_map;
};

} // namespace conversion
} // namespace core
} // namespace trtorch

