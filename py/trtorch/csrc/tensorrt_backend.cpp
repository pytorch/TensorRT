#include "torch/csrc/jit/passes/lower_graph.h"

#include "tensorrt_backend.h"
#include "tensorrt_classes.h"

#include "core/compiler.h"
#include "core/lowering/lowering.h"
#include "core/runtime/runtime.h"

namespace trtorch {
namespace backend {

c10::impl::GenericDict TensorRTBackend::compile(c10::IValue mod_val, c10::impl::GenericDict method_compile_spec) {
  auto mod = mod_val.toModule();
  auto spec = c10::impl::toTypedDict<std::string, at::IValue>(method_compile_spec);

  auto handles = c10::impl::GenericDict(
      c10::StringType::get(), c10::getCustomClassType<c10::intrusive_ptr<core::runtime::TRTEngine>>());

  for (auto it = spec.begin(), end = spec.end(); it != end; ++it) {
    auto mod_ = mod.clone();
    const auto& method_name = it->key();
    auto raw_spec = it->value().toCustomClass<trtorch::pyapi::CompileSpec>();
    LOG_DEBUG(raw_spec->stringify());
    auto cfg = raw_spec->toInternalCompileSpec();
    auto graph_and_ivals = core::lowering::Lower(mod_, method_name, cfg.lower_info);

    auto g = graph_and_ivals.first;
    auto params = graph_and_ivals.second;
    auto named_params = core::conversion::get_named_params(g->inputs(), params);

    auto convert_cfg = std::move(cfg.convert_info);
    auto device_spec = convert_cfg.engine_settings.device;
    auto device = core::runtime::CudaDevice(device_spec.gpu_id, device_spec.device_type);
    auto serialized_engine = core::conversion::ConvertBlockToEngine(g->block(), convert_cfg, named_params);
    auto engine_handle = c10::make_intrusive<core::runtime::TRTEngine>(it->key(), serialized_engine, device);
    handles.insert(method_name, at::IValue(engine_handle));
  }

  return c10::impl::toGenericDict(handles);
}

c10::impl::GenericList TensorRTBackend::execute(c10::IValue handle, c10::impl::GenericList inputs) {
  TRTORCH_ASSERT(inputs.size() > 0, "Trying to execute on empty list of arguments");
  auto engine = handle.toCustomClass<core::runtime::TRTEngine>();
  std::vector<at::Tensor> in_vec;
  for (size_t i = 0, e = inputs.size(); i < e; ++i) {
    c10::IValue val = inputs[i];
    TRTORCH_CHECK(val.isTensor(), "TensorRT currently only accepts Tensors as inputs");
    in_vec.push_back(val.toTensor());
  }
  auto outputs = core::runtime::execute_engine(in_vec, engine);
  return c10::impl::toList(c10::List<at::Tensor>(outputs));
}

namespace {
c10::IValue preprocess(
    const torch::jit::Module& mod,
    const c10::Dict<
        c10::IValue,
// this API changed between 1.9 and 1.10
#if TORCH_VERSION_MAJOR < 2 && TORCH_VERSION_MINOR < 10
        c10::IValue>& method_compile_spec
#else
        c10::IValue>& method_compile_spec,
    const torch::jit::BackendDebugHandleGenerator& generate_debug_handles
#endif
) {
  for (auto it = method_compile_spec.begin(), end = method_compile_spec.end(); it != end; ++it) {
    TRTORCH_CHECK(
        core::CheckMethodOperatorSupport(mod, it->key().toStringRef()),
        "Method " << it->key().toStringRef() << "cannot be compiled by TRTorch");
  }
  return mod._ivalue();
};

static const std::string trt("tensorrt");
static auto reg = torch::jit::backend<TensorRTBackend>(trt);
static auto preproc_reg =
    torch::jit::backend_preprocess_register(trt, torch::jit::detail::BackendPreprocessFunction(preprocess));
} // namespace

} // namespace backend
} // namespace trtorch
