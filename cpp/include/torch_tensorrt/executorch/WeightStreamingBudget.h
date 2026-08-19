#pragma once

#include <cstddef>
#include <cstdint>

namespace torch_tensorrt {
namespace executorch_backend {

// Compile spec key that carries the weight streaming budget from export into the
// delegate. Must match WEIGHT_STREAMING_BUDGET_COMPILE_SPEC_KEY on the Python
// side (py/torch_tensorrt/executorch/partitioner.py).
inline constexpr char kWeightStreamingBudgetKey[] = "weight_streaming_budget";

// Result of parsing a weight streaming budget compile spec value.
//
// The value is a non-negative decimal integer: an explicit GPU budget in bytes.
// The automatic budget is intentionally not encoded here: when no budget spec is
// present, the delegate applies TensorRT's automatic budget itself, mirroring the
// PyTorch runtimes.
//
//   valid == false -> a value was present but could not be parsed (or was
//                     negative); the caller should reject the program.
//   valid == true  -> bytes holds the parsed non-negative byte budget.
struct WsBudget {
  bool valid = false;
  int64_t bytes = 0;
};

// Parses a budget from a raw byte range. The value is NOT assumed to be NUL
// terminated; only the first nbytes bytes are read.
WsBudget parse_weight_streaming_budget(const void* value, std::size_t nbytes);

} // namespace executorch_backend
} // namespace torch_tensorrt
