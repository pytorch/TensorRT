#include <algorithm>

#include "NvInfer.h"
#include "torch/csrc/jit/frontend/function_schema_parser.h"

#include "core/util/prelude.h"
#include "core/execution/execution.h"

namespace trtorch {
namespace core {
namespace execution {

std::string slugify(std::string s) {
    std::replace(s.begin(), s.end(), '.', '_');
    return s;
}

TRTEngine::TRTEngine(std::string serialized_engine)
    : logger(std::string("[] - "),
        util::logging::get_logger().get_reportable_severity(),
        util::logging::get_logger().get_is_colored_output_on()) {
    std::string _name = "deserialized_trt";
    new (this) TRTEngine(_name, serialized_engine, empty_string);
}

TRTEngine::TRTEngine(std::vector<std::string> serialized_info)
    : logger(std::string("[] - "),
        util::logging::get_logger().get_reportable_severity(),
        util::logging::get_logger().get_is_colored_output_on()) {
    std::string _name = "deserialized_trt";
    std::string device_info = serialized_info[0];
    std::string engine_info = serialized_info[1];

    new (this) TRTEngine(_name, engine_info, device_info);
}

TRTEngine::TRTEngine(std::string mod_name, std::string serialized_engine, std::string serialized_device_info)
    : logger(std::string("[") + mod_name + std::string("_engine] - "),
        util::logging::get_logger().get_reportable_severity(),
        util::logging::get_logger().get_is_colored_output_on()) {

    device_info = serialized_device_info;

    // Deserialize device meta data if device_info is non-empty
    if (!device_info.empty())
    {
        auto cuda_device = util::deserialize_device(device_info);
        // Set CUDA device as configured in serialized meta data
        util::set_cuda_device(cuda_device);
    }

    rt = nvinfer1::createInferRuntime(logger);

    name = slugify(mod_name) + "_engine";

    cuda_engine = rt->deserializeCudaEngine(serialized_engine.c_str(), serialized_engine.size());
    // Easy way to get a unique name for each engine, maybe there is a more descriptive way (using something associated with the graph maybe)
    id = reinterpret_cast<EngineID>(cuda_engine);

    exec_ctx = cuda_engine->createExecutionContext();

    uint64_t inputs = 0;
    uint64_t outputs = 0;

    for (int64_t x = 0; x < cuda_engine->getNbBindings(); x++) {
        std::string name = cuda_engine->getBindingName(x);
        std::string idx_s = name.substr(name.find("_") + 1);
        uint64_t idx = static_cast<uint64_t>(std::stoi(idx_s));

        if(cuda_engine->bindingIsInput(x)) {
            inputs++;
            in_binding_map[x] = idx;
        } else {
            outputs++;
            out_binding_map[x] = idx;
        }
    }
    num_io = std::make_pair(inputs, outputs);

}

TRTEngine& TRTEngine::operator=(const TRTEngine& other) {
    id = other.id;
    rt = other.rt;
    cuda_engine = other.cuda_engine;
    device_info = other.device_info;
    exec_ctx = other.exec_ctx;
    num_io = other.num_io;
    return (*this);
}

TRTEngine::~TRTEngine() {
    exec_ctx->destroy();
    cuda_engine->destroy();
    rt->destroy();
}


// TODO: Implement a call method
// c10::List<at::Tensor> TRTEngine::Run(c10::List<at::Tensor> inputs) {
//     auto input_vec = inputs.vec();
//    auto output_vec = RunCudaEngine(exec_ctx, num_io, input_vec);
//
//     return c10::List<at::Tensor>(output_vec);
// }

static auto TRTORCH_UNUSED TRTEngineTSRegistrtion = torch::class_<TRTEngine>("tensorrt", "Engine")
    .def(torch::init<std::string>())
    // TODO: .def("__call__", &TRTEngine::Run)
    // TODO: .def("run", &TRTEngine::Run)
    .def_pickle(
        [](const c10::intrusive_ptr<TRTEngine>& self) -> std::vector<std::string> {
	    // Serialize TensorRT engine
	    auto serialized_trt_engine = self->cuda_engine->serialize();

	    // Adding device info related meta data to the serialized file
	    auto trt_engine = std::string((const char*)serialized_trt_engine->data(), serialized_trt_engine->size());

	    std::vector<std::string> serialize_info;
	    serialize_info.push_back(self->device_info);
	    serialize_info.push_back(trt_engine);
	    return serialize_info;
        },
         [](std::vector<std::string> seralized_info) -> c10::intrusive_ptr<TRTEngine> {
            return c10::make_intrusive<TRTEngine>(std::move(seralized_info));
        }
    );

} // namespace execution
} // namespace core
} // namespace trtorch
