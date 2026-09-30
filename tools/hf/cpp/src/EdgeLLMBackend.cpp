#include "torch_tensorrt_edge_llm/executorch/EdgeLLMBackend.h"
#include "torch_tensorrt_edge_llm/executorch/EdgeLLMBlobHeader.h"
#include "torch_tensorrt/executorch/TensorRTBlobHeader.h"

#include <executorch/runtime/platform/log.h>

namespace torch_tensorrt_edge_llm {
namespace executorch_backend {

using namespace ::executorch::runtime;
using ::torch_tensorrt::executorch_backend::TensorRTBlobHeader;

namespace {
extern const Error kEdgeLLMRegistrationResult;
}

bool EdgeLLMBackend::is_available() const {
  return kEdgeLLMRegistrationResult == Error::Ok && engine_backend_.is_available();
}

Result<DelegateHandle*> EdgeLLMBackend::init(
    BackendInitContext& context,
    FreeableBuffer* processed,
    ArrayRef<CompileSpec> compile_specs) const {
  if (!is_available()) {
    return Error::InvalidState;
  }
  if (processed == nullptr || processed->data() == nullptr) {
    return Error::InvalidArgument;
  }
  EdgeLLMBlobHeader component;
  if (!EdgeLLMBlobHeader::parse(processed->data(), processed->size(), component)) {
    ET_LOG(Error, "EdgeLLMBackend: invalid or unsupported EL01 component payload");
    return Error::InvalidProgram;
  }
  const void* nested_data = EdgeLLMBlobHeader::nested_blob_data(processed->data(), component);
  TensorRTBlobHeader engine;
  if (!TensorRTBlobHeader::parse(nested_data, component.blob_size, engine)) {
    return Error::InvalidProgram;
  }
  size_t expected_inputs = 1;
  size_t expected_outputs = 1;
  if (component.runner == "pi05_prefill") {
    expected_inputs = 3;
    expected_outputs = 3;
  } else if (component.runner == "pi05_action") {
    expected_inputs = 6;
  }
  if (engine.input_binding_names.size() != expected_inputs ||
      engine.output_binding_names.size() != expected_outputs || !engine.aliased_io.empty()) {
    ET_LOG(Error, "EdgeLLMBackend: component binding count or read-only input contract is invalid");
    return Error::InvalidProgram;
  }
  // The nested view borrows the outer payload. TensorRTBackend synchronously
  // deserializes the engine, then frees this view; only the outer buffer owns
  // storage. Release the owning payload once initialization succeeds.
  FreeableBuffer nested(nested_data, component.blob_size, nullptr);
  auto result = engine_backend_.init(context, &nested, compile_specs);
  if (result.ok()) {
    processed->Free();
  }
  return result;
}

Error EdgeLLMBackend::execute(BackendExecutionContext& context, DelegateHandle* handle, Span<EValue*> args) const {
  // Reuse Torch-TensorRT's libTorch-free engine execution, profile selection,
  // device staging, caller stream, completion events, and planned outputs.
  // PI0.5 prefix K/V are ordinary outputs and remain read-only action inputs.
  return engine_backend_.execute(context, handle, args);
}

void EdgeLLMBackend::destroy(DelegateHandle* handle) const {
  engine_backend_.destroy(handle);
}

namespace {
EdgeLLMBackend& backend() {
  static EdgeLLMBackend instance;
  return instance;
}
const Backend kEdgeLLMBackendId{"EdgeLLMBackend", &backend()};
const Error kEdgeLLMRegistrationResult = register_backend(kEdgeLLMBackendId);
}

} // namespace executorch_backend
} // namespace torch_tensorrt_edge_llm
