#include "core/runtime/runtime.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {
namespace {

// TODO: Implement a call method
// c10::List<at::Tensor> TRTEngine::Run(c10::List<at::Tensor> inputs) {
//     auto input_vec = inputs.vec();
//    auto output_vec = RunCudaEngine(exec_ctx, num_io, input_vec);
//
//     return c10::List<at::Tensor>(output_vec);
// }
static auto TORCHTRT_UNUSED TRTEngineTSRegistrtion =
    torch::class_<TRTEngine>("tensorrt", "Engine")
        .def(torch::init<std::vector<std::string>>())
        // TODO: .def("__call__", &TRTEngine::Run)
        // TODO: .def("run", &TRTEngine::Run)
        .def("__str__", &TRTEngine::to_str)
        .def_pickle(
            [](const c10::intrusive_ptr<TRTEngine>& self) -> std::vector<std::string> {
              // Serialize TensorRT engine
              auto serialized_trt_engine = self->cuda_engine->serialize();

              // Adding device info related meta data to the serialized file
              auto trt_engine = std::string((const char*)serialized_trt_engine->data(), serialized_trt_engine->size());

              std::vector<std::string> serialize_info;
              serialize_info.resize(ENGINE_IDX + 1);

              serialize_info[ABI_TARGET_IDX] = ABI_VERSION;
              serialize_info[NAME_IDX] = self->name;
              serialize_info[DEVICE_IDX] = self->device_info.serialize();
              serialize_info[ENGINE_IDX] = trt_engine;
              return serialize_info;
            },
            [](std::vector<std::string> seralized_info) -> c10::intrusive_ptr<TRTEngine> {
              return c10::make_intrusive<TRTEngine>(std::move(seralized_info));
            });

TORCH_LIBRARY(tensorrt, m) {
  m.def("execute_engine", execute_engine);
}

} // namespace
} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
