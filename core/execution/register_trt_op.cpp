#include "c10/cuda/CUDAStream.h"

#include "torch/csrc/jit/custom_operator.h"

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
        auto dims = core::util::toDimsPad(inputs[i].sizes(), 1);
        auto shape = core::util::toVec(dims);
        contig_inputs.push_back(inputs[i].to(at::kCUDA).view(shape).contiguous());
        LOG_DEBUG("In shape:" << shape);
        ctx->setBindingDimensions(i, dims);
        gpu_handles.push_back(contig_inputs.back().data_ptr());
    }

    TRTORCH_CHECK(ctx->allInputDimensionsSpecified(), "Not enough inputs provided (execution.RunCudaEngine)");

    std::vector<at::Tensor> outputs;
    for (uint64_t o = inputs.size(); o < (io.first + io.second); o++) {
        auto out_shape = ctx->getBindingDimensions(o);
        //LOG_DEBUG("Output: " << engine->getBindingName(o) << " out shape: " << out_shape);
        auto dims = core::util::toVec(out_shape);
        outputs.push_back(at::empty(dims, {at::kCUDA}).contiguous());
        gpu_handles.push_back(outputs[outputs.size() - 1].data_ptr());
    }

    // Is this the right stream?
    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(inputs[0].device().index());

    ctx->enqueueV2(gpu_handles.data(), stream, nullptr);

    return outputs;
}

c10::OperatorOptions aliasAnalysisFromSchema() {
    c10::OperatorOptions result;
    result.setAliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA);
    return result;
}


// The other way to do this is to register a generic op something liek
// trt::execute_engine(int id, Tensor input, ...) -> (Tensor...) but not sure
// how well that would work
void RegisterEngineOp(TRTEngine& engine) {
    EngineID id = engine.id;
    torch::jit::RegisterOperators jit_registry({
            torch::jit::Operator(
                engine.schema,
                [id](torch::jit::Stack& stack) {
                    LOG_DEBUG("Attempting to run engine (ID: " << std::hex << id << ")");
                    auto io = GetEngineIO(id);
                    auto num_in = io.first;
                    auto num_out = io.second;
                    // Verify calling convention (right to left or left to right)
                    std::vector<at::Tensor> inputs;
                    for (uint64_t i = 0; i < num_in; i++) {
                        at::Tensor in;
                        torch::jit::pop(stack, in);
                        inputs.insert(inputs.begin(), std::move(in));
                    }

                    auto ctx = GetExecCtx(id);
                    auto outputs = RunCudaEngine(ctx, io, inputs);
                    for (uint64_t o = 0; o < num_out; o++) {
                        torch::jit::push(stack, std::move(outputs[o]));
                    }
                    return 0;
                },
                aliasAnalysisFromSchema())
                });
}

} // namespace execution
} // namespace core
} // namespace trtorch
