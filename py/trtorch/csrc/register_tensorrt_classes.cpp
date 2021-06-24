#include "tensorrt_classes.h"

namespace trtorch {
namespace backend {
namespace {

#define ADD_FIELD_GET_SET_REGISTRATION(registry, class_name, field_name) \
  (registry).def("_set_" #field_name, &class_name::set_##field_name);    \
  (registry).def("_get_" #field_name, &class_name::get_##field_name);

void RegisterTRTCompileSpec() {
  static auto TRTORCH_UNUSED TRTInputRangeTSRegistration =
      torch::class_<trtorch::pyapi::InputRange>("tensorrt", "_InputRange")
          .def(torch::init<>())
          .def("__str__", &trtorch::pyapi::InputRange::to_str);

  ADD_FIELD_GET_SET_REGISTRATION(TRTInputRangeTSRegistration, trtorch::pyapi::InputRange, min);
  ADD_FIELD_GET_SET_REGISTRATION(TRTInputRangeTSRegistration, trtorch::pyapi::InputRange, opt);
  ADD_FIELD_GET_SET_REGISTRATION(TRTInputRangeTSRegistration, trtorch::pyapi::InputRange, max);

  static auto TRTORCH_UNUSED TRTDeviceTSRegistration = torch::class_<trtorch::pyapi::Device>("tensorrt", "_Device")
                                                           .def(torch::init<>())
                                                           .def("__str__", &trtorch::pyapi::Device::to_str);

  ADD_FIELD_GET_SET_REGISTRATION(TRTDeviceTSRegistration, trtorch::pyapi::Device, device_type);
  ADD_FIELD_GET_SET_REGISTRATION(TRTDeviceTSRegistration, trtorch::pyapi::Device, gpu_id);
  ADD_FIELD_GET_SET_REGISTRATION(TRTDeviceTSRegistration, trtorch::pyapi::Device, dla_core);
  ADD_FIELD_GET_SET_REGISTRATION(TRTDeviceTSRegistration, trtorch::pyapi::Device, allow_gpu_fallback);

  static auto TRTORCH_UNUSED TRTFallbackTSRegistration =
      torch::class_<trtorch::pyapi::TorchFallback>("tensorrt", "_TorchFallback")
          .def(torch::init<>())
          .def("__str__", &trtorch::pyapi::TorchFallback::to_str);

  ADD_FIELD_GET_SET_REGISTRATION(TRTFallbackTSRegistration, trtorch::pyapi::TorchFallback, enabled);
  ADD_FIELD_GET_SET_REGISTRATION(TRTFallbackTSRegistration, trtorch::pyapi::TorchFallback, min_block_size);
  ADD_FIELD_GET_SET_REGISTRATION(TRTFallbackTSRegistration, trtorch::pyapi::TorchFallback, forced_fallback_operators);

  static auto TRTORCH_UNUSED TRTCompileSpecTSRegistration =
      torch::class_<trtorch::pyapi::CompileSpec>("tensorrt", "CompileSpec")
          .def(torch::init<>())
          .def("_append_input_range", &trtorch::pyapi::CompileSpec::appendInputRange)
          .def("_set_device", &trtorch::pyapi::CompileSpec::setDeviceIntrusive)
          .def("_set_torch_fallback", &trtorch::pyapi::CompileSpec::setTorchFallbackIntrusive)
          .def("_set_ptq_calibrator", &trtorch::pyapi::CompileSpec::setPTQCalibratorViaHandle)
          .def("_append_input_dtypes", &trtorch::pyapi::CompileSpec::appendInputDtypes)
          .def("__str__", &trtorch::pyapi::CompileSpec::stringify);

  ADD_FIELD_GET_SET_REGISTRATION(TRTCompileSpecTSRegistration, trtorch::pyapi::CompileSpec, op_precision);
  ADD_FIELD_GET_SET_REGISTRATION(TRTCompileSpecTSRegistration, trtorch::pyapi::CompileSpec, disable_tf32);
  ADD_FIELD_GET_SET_REGISTRATION(TRTCompileSpecTSRegistration, trtorch::pyapi::CompileSpec, refit);
  ADD_FIELD_GET_SET_REGISTRATION(TRTCompileSpecTSRegistration, trtorch::pyapi::CompileSpec, debug);
  ADD_FIELD_GET_SET_REGISTRATION(TRTCompileSpecTSRegistration, trtorch::pyapi::CompileSpec, strict_types);
  ADD_FIELD_GET_SET_REGISTRATION(TRTCompileSpecTSRegistration, trtorch::pyapi::CompileSpec, capability);
  ADD_FIELD_GET_SET_REGISTRATION(TRTCompileSpecTSRegistration, trtorch::pyapi::CompileSpec, num_min_timing_iters);
  ADD_FIELD_GET_SET_REGISTRATION(TRTCompileSpecTSRegistration, trtorch::pyapi::CompileSpec, num_avg_timing_iters);
  ADD_FIELD_GET_SET_REGISTRATION(TRTCompileSpecTSRegistration, trtorch::pyapi::CompileSpec, workspace_size);
  ADD_FIELD_GET_SET_REGISTRATION(TRTCompileSpecTSRegistration, trtorch::pyapi::CompileSpec, max_batch_size);
  ADD_FIELD_GET_SET_REGISTRATION(TRTCompileSpecTSRegistration, trtorch::pyapi::CompileSpec, truncate_long_and_double);
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
