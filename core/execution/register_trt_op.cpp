#include "c10/cuda/CUDAStream.h"

#include "torch/torch.h"
#include "torch/csrc/jit/runtime/custom_operator.h"

#include "core/util/prelude.h"
#include "core/execution/execution.h"

namespace trtorch {
namespace core {
namespace execution {
std::vector<at::Tensor> RunCudaEngine(nvinfer1::IExecutionContext* ctx, std::pair<uint64_t, uint64_t> io, std::vector<at::Tensor>& inputs) {
    std::vector<void*> gpu_handles;

    std::vector<at::Tensor> contig_inputs{};
    contig_inputs.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
        TRTORCH_CHECK(inputs[i].is_cuda(), "Expected input tensors to have device cuda, found device " << inputs[i].device());
        auto expected_type = util::toATenDType(ctx->getEngine().getBindingDataType(i));
        TRTORCH_CHECK(inputs[i].dtype() == expected_type, "Expected input tensors to have type " << expected_type << ", found type " << inputs[i].dtype());
        auto dims = core::util::toDimsPad(inputs[i].sizes(), 1);
        auto shape = core::util::toVec(dims);
        contig_inputs.push_back(inputs[i].view(shape).contiguous());
        LOG_DEBUG("Input shape: " << dims);
        ctx->setBindingDimensions(i, dims);
        gpu_handles.push_back(contig_inputs.back().data_ptr());
    }

    TRTORCH_CHECK(ctx->allInputDimensionsSpecified(), "Not enough inputs provided (execution.RunCudaEngine)");

    std::vector<at::Tensor> outputs;
    for (uint64_t o = inputs.size(); o < (io.first + io.second); o++) {
        auto out_shape = ctx->getBindingDimensions(o);
        LOG_DEBUG("Output shape: " << out_shape);
        auto dims = core::util::toVec(out_shape);
        auto type = util::toATenDType(ctx->getEngine().getBindingDataType(o));
        outputs.push_back(at::empty(dims, {at::kCUDA}).to(type).contiguous());
        gpu_handles.push_back(outputs[outputs.size() - 1].data_ptr());
    }

    // Is this the right stream?
    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(inputs[0].device().index());

    ctx->enqueueV2(gpu_handles.data(), stream, nullptr);

    return outputs;
}

std::vector<at::Tensor> execute_engine(std::vector<at::Tensor> inputs, c10::intrusive_ptr<TRTEngine> engine) {
    // Verify calling convention (right to left or left to right)
    LOG_DEBUG("Attempting to run engine (ID: " << std::hex << engine->name << ")");

    auto io = engine->num_io;
    auto ctx = engine->exec_ctx;
    auto outputs = RunCudaEngine(ctx, io, inputs);

    return outputs;
}

TORCH_LIBRARY(tensorrt, m) {
  m.def("execute_engine", execute_engine);
}

} // namespace execution
} // namespace core
} // namespace trtorch
