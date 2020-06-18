#include "core/util/prelude.h"
#include "NvInfer.h"
#include "c10/cuda/CUDAStream.h"
#include "torch/csrc/jit/ir/ir.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "core/conversion/conversion.h"
#include "cuda_runtime_api.h"

#include <vector>
#include <math.h>

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

std::vector<core::conversion::InputRange> toInputRangesDynamic(std::vector<at::Tensor> ten) {
    std::vector<core::conversion::InputRange> a;

    for (auto i : ten) {
        auto opt = core::util::toVec(i.sizes());

        std::vector<int64_t> min_range(opt);
        std::vector<int64_t> max_range(opt); 

        min_range[0] = ceil(opt[0]/2.0);
        max_range[0] = 2*opt[0];

        // for (int64_t each : min_range) {
        //     std::cout << each << std::endl;
        // }
        // for (int64_t each : opt) {
        //     std::cout << each << std::endl;
        // }
        // for (int64_t each : max_range) {
        //     std::cout << each << std::endl;
        // }

        a.push_back(core::conversion::InputRange(min_range, opt, max_range));
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
        TRTORCH_CHECK(inputs[i].is_cuda(), "Expected input tensors to have device cuda, found device " << inputs[i].device());
        auto expected_type = core::util::toATenDType(ctx->getEngine().getBindingDataType(i));
        TRTORCH_CHECK(inputs[i].dtype() == expected_type, "Expected input tensors to have type " << expected_type << ", found type " << inputs[i].dtype());
        auto dims = core::util::toDimsPad(inputs[i].sizes(), 1);
        auto shape = core::util::toVec(dims);
        contig_inputs.push_back(inputs[i].view(shape).contiguous());
        LOG_DEBUG("In shape:" << shape);
        ctx->setBindingDimensions(i, dims);
        gpu_handles.push_back(contig_inputs.back().data_ptr());
    }

    TRTORCH_CHECK(ctx->allInputDimensionsSpecified(), "Not enough inputs provided (execution.RunCudaEngine)");

    std::vector<at::Tensor> outputs;
    for (int64_t o = inputs.size(); o < engine->getNbBindings(); o++) {
        auto out_shape = ctx->getBindingDimensions(o);
        LOG_DEBUG("Output: " << engine->getBindingName(o) << " out shape: " << out_shape);
        auto dims = core::util::toVec(out_shape);
        auto type = core::util::toATenDType(ctx->getEngine().getBindingDataType(o));
        outputs.push_back(at::empty(dims, {at::kCUDA}).to(type).contiguous());
        gpu_handles.push_back(outputs[outputs.size() - 1].data_ptr());
    }

    // Is this the right stream?
    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(inputs[0].device().index());

    ctx->enqueueV2(gpu_handles.data(), stream, nullptr);

    stream.synchronize();
    return outputs;
}

std::vector<at::Tensor> RunGraphEngine(std::shared_ptr<torch::jit::Graph>& g,
                                       core::conversion::GraphParams& named_params,
                                       std::vector<at::Tensor> inputs) {
    LOG_DEBUG("Running TRT version");
    auto in =  toInputRanges(inputs);
    auto info = core::conversion::ConversionInfo(in);
    info.engine_settings.workspace_size = 1 << 20;
    std::string eng = core::conversion::ConvertBlockToEngine(g->block(), info, named_params);
    return RunEngine(eng, inputs);
}

std::vector<at::Tensor> RunGraphEngineDynamic(std::shared_ptr<torch::jit::Graph>& g,
                                              core::conversion::GraphParams& named_params,
                                              std::vector<at::Tensor> inputs) {
    LOG_DEBUG("Running TRT version");
    auto in = toInputRangesDynamic(inputs);
    auto info = core::conversion::ConversionInfo(in);
    info.engine_settings.workspace_size = 1 << 20;
    std::string eng = core::conversion::ConvertBlockToEngine(g->block(), info, named_params);
    return RunEngine(eng, inputs);
}

} // namespace util
} // namespace tests
} // namespace trtorch
