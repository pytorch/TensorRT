#include "c10/cuda/CUDAStream.h"

#include "torch/torch.h"
#include "torch/csrc/jit/runtime/custom_operator.h"

#include "core/util/prelude.h"
#include "core/runtime/runtime.h"

namespace trtorch {
namespace core {
namespace runtime {

std::vector<at::Tensor> execute_engine(std::vector<at::Tensor> inputs, c10::intrusive_ptr<TRTEngine> compiled_engine) {
    LOG_DEBUG("Attempting to run engine (ID: " << compiled_engine->name << ")");
    std::vector<void*> gpu_handles;

    std::vector<at::Tensor> contig_inputs{};
    contig_inputs.reserve(inputs.size());

    for (size_t i = 0; i < inputs.size(); i++) {
        uint64_t pyt_idx = compiled_engine->in_binding_map[i];
        TRTORCH_CHECK(inputs[pyt_idx].is_cuda(), "Expected input tensors to have device cuda, found device " << inputs[pyt_idx].device());
        auto expected_type = util::toATenDType(compiled_engine->exec_ctx->getEngine().getBindingDataType(i));
        TRTORCH_CHECK(inputs[pyt_idx].dtype() == expected_type, "Expected input tensors to have type " << expected_type << ", found type " << inputs[pyt_idx].dtype());
        auto dims = core::util::toDimsPad(inputs[pyt_idx].sizes(), 1);
        auto shape = core::util::toVec(dims);
        contig_inputs.push_back(inputs[pyt_idx].view(shape).contiguous());
        LOG_DEBUG("Input shape: " << dims);
        compiled_engine->exec_ctx->setBindingDimensions(i, dims);
        gpu_handles.push_back(contig_inputs.back().data_ptr());
    }

    TRTORCH_CHECK(compiled_engine->exec_ctx->allInputDimensionsSpecified(), "Not enough inputs provided (execution.RunCudaEngine)");

    std::vector<at::Tensor> outputs(compiled_engine->num_io.second);
    for (size_t o = inputs.size(); o < (compiled_engine->num_io.first + compiled_engine->num_io.second); o++) {
        uint64_t pyt_idx = compiled_engine->out_binding_map[o];
        auto out_shape = compiled_engine->exec_ctx->getBindingDimensions(o);
        LOG_DEBUG("Output shape: " << out_shape);
        auto dims = core::util::toVec(out_shape);
        auto type = util::toATenDType(compiled_engine->exec_ctx->getEngine().getBindingDataType(o));
        outputs[pyt_idx] = std::move(at::empty(dims, {at::kCUDA}).to(type).contiguous());
        gpu_handles.push_back(outputs[pyt_idx].data_ptr());
    }

    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(inputs[0].device().index());
    compiled_engine->exec_ctx->enqueueV2(gpu_handles.data(), stream, nullptr);

    return outputs;
}

TORCH_LIBRARY(tensorrt, m) {
  m.def("execute_engine", execute_engine);
}

} // namespace execution
} // namespace core
} // namespace trtorch
