#include <codecvt>

#include "core/runtime/Platform.h"
#include "core/runtime/RuntimeSettings.h"
#include "core/runtime/runtime.h"
#include "core/util/macros.h"

// serialize_bindings / base64_encode / base64_decode are defined in
// runtime_utils.cpp so the ExecuTorch backend can link them without
// pulling in the torch::class_ registration below.

namespace torch_tensorrt {
namespace core {
namespace runtime {

namespace {

// Register `RuntimeCacheHandle` as a torchbind class so Python can pass the same
// underlying `IRuntimeCache` to both Python and C++ engine backends. File I/O on
// the handle is the Python side's responsibility; the C++ struct only holds the
// shared_ptr and an informational path string. The method bodies (and the
// `#ifdef TRT_MAJOR_RTX` they entail) live in RuntimeSettings.cpp -- this file
// is registration-only.
static auto TORCHTRT_UNUSED RuntimeCacheHandleRegistration =
    torch::class_<RuntimeCacheHandle>("tensorrt", "RuntimeCacheHandle")
        .def(torch::init<std::string>())
        .def_readwrite("path", &RuntimeCacheHandle::path)
        .def("serialize", &RuntimeCacheHandle::serialize)
        .def("deserialize", &RuntimeCacheHandle::deserialize)
        .def("has_cache", &RuntimeCacheHandle::has_cache)
        // ``def_pickle`` registers ``__getstate__`` / ``__setstate__`` so the
        // handle can survive ``deepcopy`` / ``torch.export.save`` paths that
        // walk Python attributes (e.g. ``TorchTensorRTModule._implicit_cache_handle``).
        // We persist only the ``path`` string: the underlying ``IRuntimeCache``
        // is GPU-side state that can't cross a process boundary anyway, and
        // ``_resolve_runtime_cache`` re-warms from disk on the deserialized
        // path through the standard load -> pending_warm_bytes flow.
        .def_pickle(
            // __getstate__
            [](c10::intrusive_ptr<RuntimeCacheHandle> const& self) -> std::string { return self->path; },
            // __setstate__
            [](std::string path) -> c10::intrusive_ptr<RuntimeCacheHandle> {
              return c10::make_intrusive<RuntimeCacheHandle>(std::move(path));
            });

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
        .def("__repr__", &TRTEngine::to_str)
        .def("__obj_flatten__", &TRTEngine::__obj_flatten__)
        .def("enable_profiling", &TRTEngine::enable_profiling)
        .def("set_profile_format", &TRTEngine::set_profile_format)
        .def("disable_profiling", &TRTEngine::disable_profiling)
        .def_readwrite("profile_path_prefix", &TRTEngine::profile_path_prefix)
        .def("dump_engine_layer_info_to_file", &TRTEngine::dump_engine_layer_info_to_file)
        .def("dump_engine_layer_info", &TRTEngine::dump_engine_layer_info)
        .def("get_engine_layer_info", &TRTEngine::get_engine_layer_info)
        .def("get_serialized_metadata", &TRTEngine::get_serialized_metadata)
        .def("infer_outputs", &TRTEngine::infer_outputs)
        .def("reset_captured_graph", &TRTEngine::reset_captured_graph)
        .def("set_output_tensors_as_unowned", &TRTEngine::set_output_tensors_as_unowned)
        .def("are_output_tensors_unowned", &TRTEngine::are_output_tensors_unowned)
        // Lambda wrapper because torchbind's ``def`` template lacks a
        // ``const noexcept`` member-function specialization; routing through a
        // plain function pointer would force us to drop the ``noexcept`` on
        // ``num_execution_contexts_created`` itself.
        .def(
            "num_execution_contexts_created",
            [](const c10::intrusive_ptr<TRTEngine>& self) -> int64_t { return self->num_execution_contexts_created(); })
        .def(
            "use_dynamically_allocated_resources",
            [](const c10::intrusive_ptr<TRTEngine>& self, bool dynamic) -> void {
              self->set_resource_allocation_strategy(
                  dynamic ? TRTEngine::ResourceAllocationStrategy::kDynamic
                          : TRTEngine::ResourceAllocationStrategy::kStatic);
            })
        .def(
            "update_runtime_settings",
            [](const c10::intrusive_ptr<TRTEngine>& self,
               int64_t dynamic_shapes_kernel_specialization_strategy,
               int64_t cuda_graph_strategy,
               c10::optional<c10::intrusive_ptr<RuntimeCacheHandle>> runtime_cache) -> void {
              // Strategies cross the Py->C++ boundary as ints; ``c10::optional``
              // lets TorchBind accept Python ``None`` for the cache (translated
              // to a possibly-null ``intrusive_ptr``).
              RuntimeSettings rs;
              rs.dynamic_shapes_kernel_specialization_strategy =
                  DynamicShapesKernelSpecializationStrategy::from_underlying(
                      dynamic_shapes_kernel_specialization_strategy);
              rs.cuda_graph_strategy = CudaGraphStrategy::from_underlying(cuda_graph_strategy);
              rs.runtime_cache = runtime_cache.has_value() ? std::move(*runtime_cache) : nullptr;
              (void)self->runtime_settings(std::move(rs));
            })
        .def_readwrite("use_pre_allocated_outputs", &TRTEngine::use_pre_allocated_outputs)
        .def_readwrite("pre_allocated_outputs", &TRTEngine::pre_allocated_outputs)
        .def_readwrite("use_output_allocator_outputs", &TRTEngine::use_output_allocator_outputs)
        .def_property(
            "device_memory_budget",
            &TRTEngine::get_device_memory_budget,
            &TRTEngine::set_device_memory_budget)
        .def_property("streamable_device_memory_budget", &TRTEngine::get_streamable_device_memory_budget)
        .def_property("automatic_device_memory_budget", &TRTEngine::get_automatic_device_memory_budget)
        .def_readonly("requires_native_multidevice", &TRTEngine::requires_native_multidevice)
        .def_readonly("rank", &TRTEngine::rank)
        .def_readonly("world_size", &TRTEngine::world_size)
#ifdef ENABLE_TRT_NCCL_COLLECTIVES
        .def(
            "set_group_name",
            [](c10::intrusive_ptr<TRTEngine> self, std::string group_name) {
              // Only reset nccl_initialized when the group actually changes.
              // Re-pinning the same group should be a no-op — calling
              // setCommunicator() on an exec_ctx that already has one causes
              // a TRT API error ("existing communicator must be null").
              if (self->group_name != group_name) {
                self->group_name = group_name;
                self->nccl_initialized = false;
                LOG_DEBUG("TRTEngine group_name changed to '" << group_name << "'");
              }
            })
        .def("bind_nccl_comm", [](c10::intrusive_ptr<TRTEngine> self) { self->bind_nccl_comm(); })
        .def("release_nccl_comm", [](c10::intrusive_ptr<TRTEngine> self) { self->release_nccl_comm(); })
        .def_readonly("nccl_initialized", &TRTEngine::nccl_initialized)
#else
        .def(
            "set_group_name",
            [](c10::intrusive_ptr<TRTEngine> self, std::string group_name) {
              LOG_ERROR(
                  "This build does not support MultiDevice TensorRT (ENABLE_TRT_NCCL_COLLECTIVES is OFF); set_group_name is a no-op");
            })
        .def(
            "bind_nccl_comm",
            [](c10::intrusive_ptr<TRTEngine> self) {
              LOG_ERROR(
                  "This build does not support MultiDevice TensorRT (ENABLE_TRT_NCCL_COLLECTIVES is OFF); bind_nccl_comm is a no-op");
            })
        .def(
            "release_nccl_comm",
            [](c10::intrusive_ptr<TRTEngine> self) {
              LOG_ERROR(
                  "This build does not support MultiDevice TensorRT (ENABLE_TRT_NCCL_COLLECTIVES is OFF); release_nccl_comm is a no-op");
            })
        .def_readonly("nccl_initialized", &TRTEngine::_native_nccl_support)
#endif
        .def_pickle(
            [](const c10::intrusive_ptr<TRTEngine>& self) -> std::vector<std::string> { return self->serialize(); },
            [](std::vector<std::string> serialized_info) -> c10::intrusive_ptr<TRTEngine> {
              serialized_info[ENGINE_IDX] = base64_decode(serialized_info[ENGINE_IDX]);
              LOG_DEBUG(
                  "Deserialized resource allocation strategy: "
                  << (static_cast<bool>(std::stoi(serialized_info[RESOURCE_ALLOCATION_STRATEGY_IDX])) ? "Dynamic"
                                                                                                      : "Static"));
              TRTEngine::verify_serialization_fmt(serialized_info);
              return c10::make_intrusive<TRTEngine>(serialized_info);
            });

TORCH_LIBRARY(tensorrt, m) {
  m.def("execute_engine(Tensor[] input_tensors, __torch__.torch.classes.tensorrt.Engine engine) -> Tensor[]");
  m.def("SERIALIZED_ENGINE_BINDING_DELIM", []() -> std::string { return std::string(1, TRTEngine::BINDING_DELIM); });
  m.def("SERIALIZED_RT_DEVICE_DELIM", []() -> std::string { return DEVICE_INFO_DELIM; });
  m.def("ABI_VERSION", []() -> std::string { return ABI_VERSION; });
  m.def("get_multi_device_safe_mode", []() -> bool { return MULTI_DEVICE_SAFE_MODE; });
  m.def("set_multi_device_safe_mode", [](bool multi_device_safe_mode) -> void {
    MULTI_DEVICE_SAFE_MODE = multi_device_safe_mode;
  });
  m.def("get_cudagraphs_mode", []() -> int64_t { return CUDAGRAPHS_MODE; });
  m.def("set_cudagraphs_mode", [](int64_t cudagraphs_mode) -> void {
    CUDAGRAPHS_MODE = CudaGraphsMode(cudagraphs_mode);
  });
  m.def("set_logging_level", [](int64_t level) -> void {
    util::logging::get_logger().set_reportable_log_level(util::logging::LogLevel(level));
  });
  m.def(
      "get_logging_level", []() -> int64_t { return int64_t(util::logging::get_logger().get_reportable_log_level()); });
  m.def("ABI_TARGET_IDX", []() -> int64_t { return ABI_TARGET_IDX; });
  m.def("NAME_IDX", []() -> int64_t { return NAME_IDX; });
  m.def("DEVICE_IDX", []() -> int64_t { return DEVICE_IDX; });
  m.def("ENGINE_IDX", []() -> int64_t { return ENGINE_IDX; });
  m.def("INPUT_BINDING_NAMES_IDX", []() -> int64_t { return INPUT_BINDING_NAMES_IDX; });
  m.def("OUTPUT_BINDING_NAMES_IDX", []() -> int64_t { return OUTPUT_BINDING_NAMES_IDX; });
  m.def("HW_COMPATIBLE_IDX", []() -> int64_t { return HW_COMPATIBLE_IDX; });
  m.def("SERIALIZED_METADATA_IDX", []() -> int64_t { return SERIALIZED_METADATA_IDX; });
  m.def("TARGET_PLATFORM_IDX", []() -> int64_t { return TARGET_PLATFORM_IDX; });
  m.def("REQUIRES_OUTPUT_ALLOCATOR_IDX", []() -> int64_t { return REQUIRES_OUTPUT_ALLOCATOR_IDX; });
  m.def("SERIALIZATION_LEN", []() -> int64_t { return SERIALIZATION_LEN; });
  m.def("RESOURCE_ALLOCATION_STRATEGY_IDX", []() -> int64_t { return RESOURCE_ALLOCATION_STRATEGY_IDX; });
  m.def("REQUIRES_NATIVE_MULTIDEVICE_IDX", []() -> int64_t { return REQUIRES_NATIVE_MULTIDEVICE_IDX; });
  m.def("NATIVE_TRT_COLLECTIVES_AVAIL", []() -> bool {
#ifdef ENABLE_TRT_NCCL_COLLECTIVES
    return true;
#else
    return false;
#endif
  });
  m.def("_platform_linux_x86_64", []() -> std::string {
    auto it = get_platform_name_map().find(Platform::PlatformEnum::kLINUX_X86_64);
    return it->second;
  });
  m.def("_platform_linux_aarch64", []() -> std::string {
    auto it = get_platform_name_map().find(Platform::PlatformEnum::kLINUX_AARCH64);
    return it->second;
  });
  m.def("_platform_win_x86_64", []() -> std::string {
    auto it = get_platform_name_map().find(Platform::PlatformEnum::kWIN_X86_64);
    return it->second;
  });
  m.def("_platform_unknown", []() -> std::string {
    auto it = get_platform_name_map().find(Platform::PlatformEnum::kUNKNOWN);
    return it->second;
  });
  m.def("get_current_platform", []() -> std::string {
    auto it = get_platform_name_map().find(get_current_platform()._platform);
    return it->second;
  });
}

TORCH_LIBRARY_IMPL(tensorrt, CompositeExplicitAutograd, m) {
  m.impl("execute_engine", execute_engine);
}

} // namespace
} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
