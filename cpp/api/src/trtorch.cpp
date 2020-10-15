#include "torch/csrc/jit/api/module.h"

#include "core/util/prelude.h"
#include "core/compiler.h"

#include "trtorch/trtorch.h"

namespace trtorch {

// Defined in extra_info.cpp
core::ExtraInfo to_internal_extra_info(ExtraInfo external);

bool CheckMethodOperatorSupport(const torch::jit::script::Module& module,
                                std::string method_name) {
    return core::CheckMethodOperatorSupport(module, method_name);
}

std::string ConvertGraphToTRTEngine(const torch::jit::script::Module& module,
                                    std::string method_name, ExtraInfo info) {
    LOG_DEBUG(get_build_info());
    // Want to export a much simpler (non TRT header dependent) API so doing the
    // type conversion here
    return std::move(core::ConvertGraphToTRTEngine(module, method_name, to_internal_extra_info(info)));
}

torch::jit::script::Module CompileGraph(const torch::jit::script::Module& module, ExtraInfo info) {
    LOG_DEBUG(get_build_info());
    // Want to export a much simpler (non TRT header dependent) API so doing the
    // type conversion here
    return core::CompileGraph(module, to_internal_extra_info(info));
}

std::string get_build_info() {
    auto info = core::util::get_build_info();
    return std::string("TRTorch Version: ") + TRTORCH_VERSION + '\n' + info;
}

void dump_build_info() {
    std::cout << get_build_info() << std::endl;
}

void set_device(const int gpu_id) {
    // Want to export a much simpler (non CUDA header dependent) API
    core::set_device(gpu_id);
}

} // namespace trtorch
