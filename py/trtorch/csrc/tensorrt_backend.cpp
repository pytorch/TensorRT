#include "torch/csrc/jit/passes/lower_graph.h"

#include "tensorrt_backend.h"
#include "tensorrt_classes.h"

#include "core/compiler.h"
#include "core/lowering/lowering.h"
#include "core/runtime/runtime.h"

namespace trtorch {
namespace backend {

c10::IValue TensorRTBackend::preprocess(c10::IValue mod, c10::impl::GenericDict method_compile_spec) {
  auto mod_ = mod.toModule();
  LOG_DEBUG("Placing module in eval mode if not already");
  mod_.eval();
  mod_ = core::lowering::LowerModule(mod_);

  auto spec = c10::impl::toTypedDict<std::string, at::IValue>(method_compile_spec);

  for (auto it = spec.begin(), end = spec.end(); it != end; ++it) {
    TRTORCH_CHECK(
        core::CheckMethodOperatorSupport(mod.toModule(), it->key()),
        "Method " << it->key() << "cannot be compiled by TRTorch");
  }

  for (auto it = spec.begin(), end = spec.end(); it != end; ++it) {
    const auto& method_name = it->key();
    auto method = mod_.get_method(method_name);
    auto graph = method.graph();
    core::lowering::LowerGraph(graph);
  }

  return mod_._ivalue();
}

c10::impl::GenericDict TensorRTBackend::compile(c10::IValue processed_mod, c10::impl::GenericDict method_compile_spec) {
  auto mod = processed_mod.toModule();
  auto spec = c10::impl::toTypedDict<std::string, at::IValue>(method_compile_spec);

  auto handles = c10::impl::GenericDict(
      c10::StringType::get(), c10::getCustomClassType<c10::intrusive_ptr<core::runtime::TRTEngine>>());

  for (auto it = spec.begin(), end = spec.end(); it != end; ++it) {
    const auto& method_name = it->key();
    auto method = mod.get_method(method_name);
    auto g = method.graph();

    auto raw_spec = it->value().toGenericDict().at(it->key()).toCustomClass<trtorch::pyapi::CompileSpec>();
    LOG_DEBUG(raw_spec->stringify());
    auto cfg = raw_spec->toInternalCompileSpec();
    auto convert_cfg = std::move(cfg.convert_info);
    auto graph_and_ivalues = torch::jit::LowerGraph(*g, mod._ivalue());

    g = graph_and_ivalues.first;
    auto params = graph_and_ivalues.second;
    auto named_params = core::conversion::get_named_params(g->inputs(), params);

    auto serialized_engine = core::conversion::ConvertBlockToEngine(g->block(), convert_cfg, named_params);
    auto engine_handle = c10::make_intrusive<core::runtime::TRTEngine>(it->key(), serialized_engine);
    handles.insert(method.name(), at::IValue(engine_handle));
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
static auto reg = torch::jit::backend<TensorRTBackend>("tensorrt");
}

} // namespace backend
} // namespace trtorch