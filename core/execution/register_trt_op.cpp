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

namespace {
c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

// Switched to a global operator because op implementations need to be non-capturing lambdas in PYT 1.5.0+
torch::jit::RegisterOperators jit_registry({
    torch::jit::Operator(
        "trt::execute_engine(Tensor[] inputs, __torch__.torch.classes.tensorrt.Engine engine) -> Tensor[]",
        [](torch::jit::Stack& stack) -> int {
            // Verify calling convention (right to left or left to right)
            auto engine = torch::jit::pop(stack).toCustomClass<TRTEngine>();
            LOG_DEBUG("Attempting to run engine (ID: " << std::hex << engine->name << ")");

            auto inputs = torch::jit::pop(stack).toTensorVector();

            auto io = engine->num_io;

            auto ctx = engine->exec_ctx;
            auto outputs = RunCudaEngine(ctx, io, inputs);
            torch::jit::push(stack, std::move(outputs));
            return 0;
        },
        aliasAnalysisFromSchema())
    });

} // namespace
} // namespace execution
} // namespace core
} // namespace trtorch
