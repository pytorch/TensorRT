#include "tensorrt_classes.h"

namespace trtorch {
namespace backend {
namespace {
void RegisterTRTCompileSpec() {
#define ADD_FIELD_GET_SET_REGISTRATION(registry, class_name, field_name) \
  (registry).def("set_" #field_name, &class_name::set_##field_name);     \
  (registry).def("get_" #field_name, &class_name::get_##field_name);

  static auto TRTORCH_UNUSED TRTInputRangeTSRegistrtion =
      torch::class_<trtorch::pyapi::InputRange>("tensorrt", "InputRange").def(torch::init<>());

  ADD_FIELD_GET_SET_REGISTRATION(TRTInputRangeTSRegistrtion, trtorch::pyapi::InputRange, min);
  ADD_FIELD_GET_SET_REGISTRATION(TRTInputRangeTSRegistrtion, trtorch::pyapi::InputRange, opt);
  ADD_FIELD_GET_SET_REGISTRATION(TRTInputRangeTSRegistrtion, trtorch::pyapi::InputRange, max);

  static auto TRTORCH_UNUSED TRTCompileSpecTSRegistrtion =
      torch::class_<trtorch::pyapi::CompileSpec>("tensorrt", "CompileSpec")
          .def(torch::init<>())
          .def("append_input_range", &trtorch::pyapi::CompileSpec::appendInputRange)
          .def("__str__", &trtorch::pyapi::CompileSpec::stringify);

  ADD_FIELD_GET_SET_REGISTRATION(TRTCompileSpecTSRegistrtion, trtorch::pyapi::CompileSpec, op_precision);
  ADD_FIELD_GET_SET_REGISTRATION(TRTCompileSpecTSRegistrtion, trtorch::pyapi::CompileSpec, refit);
  ADD_FIELD_GET_SET_REGISTRATION(TRTCompileSpecTSRegistrtion, trtorch::pyapi::CompileSpec, debug);
  ADD_FIELD_GET_SET_REGISTRATION(TRTCompileSpecTSRegistrtion, trtorch::pyapi::CompileSpec, strict_types);
  ADD_FIELD_GET_SET_REGISTRATION(TRTCompileSpecTSRegistrtion, trtorch::pyapi::CompileSpec, allow_gpu_fallback);
  ADD_FIELD_GET_SET_REGISTRATION(TRTCompileSpecTSRegistrtion, trtorch::pyapi::CompileSpec, device);
  ADD_FIELD_GET_SET_REGISTRATION(TRTCompileSpecTSRegistrtion, trtorch::pyapi::CompileSpec, capability);
  ADD_FIELD_GET_SET_REGISTRATION(TRTCompileSpecTSRegistrtion, trtorch::pyapi::CompileSpec, num_min_timing_iters);
  ADD_FIELD_GET_SET_REGISTRATION(TRTCompileSpecTSRegistrtion, trtorch::pyapi::CompileSpec, num_avg_timing_iters);
  ADD_FIELD_GET_SET_REGISTRATION(TRTCompileSpecTSRegistrtion, trtorch::pyapi::CompileSpec, workspace_size);
  ADD_FIELD_GET_SET_REGISTRATION(TRTCompileSpecTSRegistrtion, trtorch::pyapi::CompileSpec, max_batch_size);
}

struct TRTTSRegistrations {
  TRTTSRegistrations() {
    RegisterTRTCompileSpec();
  }
};

static TRTTSRegistrations register_trt_classes = TRTTSRegistrations();
} // namespace
} // namespace backend
} // namespace trtorch
