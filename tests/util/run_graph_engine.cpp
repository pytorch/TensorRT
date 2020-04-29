#include "core/util/prelude.h"
#include "NvInfer.h"
#include "c10/cuda/CUDAStream.h"
#include "torch/csrc/jit/ir/ir.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "core/conversion/conversion.h"
#include "cuda_runtime_api.h"

namespace trtorch {
namespace tests {
namespace util {

std::vector<core::conversion::InputRange> toInputRanges(std::vector<at::Tensor> ten) {
    std::vector<core::conversion::InputRange> a;
    for (auto i : ten) {
        a.push_back(core::conversion::InputRange(core::util::toVec(i.sizes())));
    }
    return std::move(a);
}

std::vector<at::Tensor> RunEngine(std::string& eng, std::vector<at::Tensor> inputs) {
    auto rt = nvinfer1::createInferRuntime(core::util::logging::get_logger());
    auto engine = rt->deserializeCudaEngine(eng.c_str(), eng.size());
    auto ctx = engine->createExecutionContext();

    std::vector<void*> gpu_handles;

    std::vector<at::Tensor> contig_inputs{};
    contig_inputs.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
        auto dims = core::util::toDimsPad(inputs[i].sizes(), 1);
        auto shape = core::util::toVec(dims);
        contig_inputs.push_back(inputs[i].to(at::kCUDA).view(shape).contiguous());
        LOG_DEBUG("In shape:" << shape);
        ctx->setBindingDimensions(i, dims);
        gpu_handles.push_back(contig_inputs.back().data_ptr());
    }

    if (!ctx->allInputDimensionsSpecified()) {
        LOG_ERROR("Not enough inputs provided (tests.runEngine)");
        return {};
    }

    std::vector<at::Tensor> outputs;
    for (int o = inputs.size(); o < engine->getNbBindings(); o++) {
        auto out_shape = ctx->getBindingDimensions(o);
        LOG_DEBUG("Output: " << engine->getBindingName(o) << " out shape: " << out_shape);
        auto dims = core::util::toVec(out_shape);
        outputs.push_back(at::empty(dims, {at::kCUDA}).contiguous());
        gpu_handles.push_back(outputs[outputs.size() - 1].data_ptr());
    }


    c10::cuda::CUDAStream stream = c10::cuda::getStreamFromPool(true, 0);

    ctx->enqueueV2(gpu_handles.data(), stream, nullptr);
    stream.synchronize();

    return outputs;
}

std::vector<at::Tensor> RunGraphEngine(std::shared_ptr<torch::jit::Graph>& g,
                                       core::conversion::GraphParams& named_params,
                                       std::vector<at::Tensor> inputs) {
    LOG_DEBUG("Running TRT version");
    auto in =  toInputRanges(inputs);
    std::string eng = core::conversion::ConvertBlockToEngine(g->block(), in, named_params);
    return RunEngine(eng, inputs);
}

} // namespace util
} // namespace tests
} // namespace trtorch
