#include "tensorrt_classes.h"

namespace torch_tensorrt {
namespace torchscript {
namespace backend {
namespace {

#define ADD_FIELD_GET_SET_REGISTRATION(registry, class_name, field_name) \
  (registry).def("_set_" #field_name, &class_name::set_##field_name);    \
  (registry).def("_get_" #field_name, &class_name::get_##field_name);

void RegisterTRTCompileSpec() {
  static auto TORCHTRT_UNUSED TRTInputRangeTSRegistration =
      torch::class_<torch_tensorrt::pyapi::Input>("tensorrt", "_Input")
          .def(torch::init<>())
          .def("__str__", &torch_tensorrt::pyapi::Input::to_str);

  ADD_FIELD_GET_SET_REGISTRATION(TRTInputRangeTSRegistration, torch_tensorrt::pyapi::Input, min);
  ADD_FIELD_GET_SET_REGISTRATION(TRTInputRangeTSRegistration, torch_tensorrt::pyapi::Input, opt);
  ADD_FIELD_GET_SET_REGISTRATION(TRTInputRangeTSRegistration, torch_tensorrt::pyapi::Input, max);
  ADD_FIELD_GET_SET_REGISTRATION(TRTInputRangeTSRegistration, torch_tensorrt::pyapi::Input, dtype);
  ADD_FIELD_GET_SET_REGISTRATION(TRTInputRangeTSRegistration, torch_tensorrt::pyapi::Input, format);
  ADD_FIELD_GET_SET_REGISTRATION(TRTInputRangeTSRegistration, torch_tensorrt::pyapi::Input, input_is_dynamic);
  ADD_FIELD_GET_SET_REGISTRATION(TRTInputRangeTSRegistration, torch_tensorrt::pyapi::Input, explicit_set_dtype);

  static auto TORCHTRT_UNUSED TRTInputSignatureTSRegistration =
      torch::class_<torch_tensorrt::pyapi::InputSignature>("tensorrt", "_InputSignature")
          .def(torch::init<>())
          .def(
              "_set_signature_ivalue_torchbind",
              [](const c10::intrusive_ptr<torch_tensorrt::pyapi::InputSignature>& self, torch::jit::IValue ival) {
                self->signature_ivalue = ival;
              })
          .def("__str__", &torch_tensorrt::pyapi::InputSignature::to_str);

  ADD_FIELD_GET_SET_REGISTRATION(
      TRTInputSignatureTSRegistration, torch_tensorrt::pyapi::InputSignature, signature_ivalue);

  static auto TORCHTRT_UNUSED TRTDeviceTSRegistration =
      torch::class_<torch_tensorrt::pyapi::Device>("tensorrt", "_Device")
          .def(torch::init<>())
          .def("__str__", &torch_tensorrt::pyapi::Device::to_str);

  ADD_FIELD_GET_SET_REGISTRATION(TRTDeviceTSRegistration, torch_tensorrt::pyapi::Device, device_type);
  ADD_FIELD_GET_SET_REGISTRATION(TRTDeviceTSRegistration, torch_tensorrt::pyapi::Device, gpu_id);
  ADD_FIELD_GET_SET_REGISTRATION(TRTDeviceTSRegistration, torch_tensorrt::pyapi::Device, dla_core);
  ADD_FIELD_GET_SET_REGISTRATION(TRTDeviceTSRegistration, torch_tensorrt::pyapi::Device, allow_gpu_fallback);

  static auto TORCHTRT_UNUSED TRTFallbackTSRegistration =
      torch::class_<torch_tensorrt::pyapi::TorchFallback>("tensorrt", "_TorchFallback")
          .def(torch::init<>())
          .def("__str__", &torch_tensorrt::pyapi::TorchFallback::to_str);

  ADD_FIELD_GET_SET_REGISTRATION(TRTFallbackTSRegistration, torch_tensorrt::pyapi::TorchFallback, enabled);
  ADD_FIELD_GET_SET_REGISTRATION(TRTFallbackTSRegistration, torch_tensorrt::pyapi::TorchFallback, min_block_size);
  ADD_FIELD_GET_SET_REGISTRATION(
      TRTFallbackTSRegistration, torch_tensorrt::pyapi::TorchFallback, forced_fallback_operators);
  ADD_FIELD_GET_SET_REGISTRATION(
      TRTFallbackTSRegistration, torch_tensorrt::pyapi::TorchFallback, forced_fallback_modules);

  static auto TORCHTRT_UNUSED TRTCompileSpecTSRegistration =
      torch::class_<torch_tensorrt::pyapi::CompileSpec>("tensorrt", "CompileSpec")
          .def(torch::init<>())
          .def("_append_input", &torch_tensorrt::pyapi::CompileSpec::appendInput)
          .def("_set_input_signature", &torch_tensorrt::pyapi::CompileSpec::setInputSignature)
          .def("_set_precisions", &torch_tensorrt::pyapi::CompileSpec::setPrecisions)
          .def("_set_device", &torch_tensorrt::pyapi::CompileSpec::setDeviceIntrusive)
          .def("_set_torch_fallback", &torch_tensorrt::pyapi::CompileSpec::setTorchFallbackIntrusive)
          .def("_set_ptq_calibrator", &torch_tensorrt::pyapi::CompileSpec::setPTQCalibratorViaHandle)
          .def("__str__", &torch_tensorrt::pyapi::CompileSpec::stringify);

  ADD_FIELD_GET_SET_REGISTRATION(TRTCompileSpecTSRegistration, torch_tensorrt::pyapi::CompileSpec, sparse_weights);
  ADD_FIELD_GET_SET_REGISTRATION(TRTCompileSpecTSRegistration, torch_tensorrt::pyapi::CompileSpec, disable_tf32);
  ADD_FIELD_GET_SET_REGISTRATION(TRTCompileSpecTSRegistration, torch_tensorrt::pyapi::CompileSpec, refit);
  ADD_FIELD_GET_SET_REGISTRATION(TRTCompileSpecTSRegistration, torch_tensorrt::pyapi::CompileSpec, debug);
  ADD_FIELD_GET_SET_REGISTRATION(TRTCompileSpecTSRegistration, torch_tensorrt::pyapi::CompileSpec, capability);
  ADD_FIELD_GET_SET_REGISTRATION(
      TRTCompileSpecTSRegistration, torch_tensorrt::pyapi::CompileSpec, num_avg_timing_iters);
  ADD_FIELD_GET_SET_REGISTRATION(TRTCompileSpecTSRegistration, torch_tensorrt::pyapi::CompileSpec, workspace_size);
  ADD_FIELD_GET_SET_REGISTRATION(TRTCompileSpecTSRegistration, torch_tensorrt::pyapi::CompileSpec, dla_sram_size);
  ADD_FIELD_GET_SET_REGISTRATION(TRTCompileSpecTSRegistration, torch_tensorrt::pyapi::CompileSpec, dla_local_dram_size);
  ADD_FIELD_GET_SET_REGISTRATION(
      TRTCompileSpecTSRegistration, torch_tensorrt::pyapi::CompileSpec, dla_global_dram_size);
  ADD_FIELD_GET_SET_REGISTRATION(
      TRTCompileSpecTSRegistration, torch_tensorrt::pyapi::CompileSpec, truncate_long_and_double);
}

struct TRTTSRegistrations {
  TRTTSRegistrations() {
    RegisterTRTCompileSpec();
  }
};

static TRTTSRegistrations register_trt_classes = TRTTSRegistrations();
} // namespace
} // namespace backend
} // namespace torchscript
} // namespace torch_tensorrt
