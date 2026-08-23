#include "torch_tensorrt/executorch/WeightStreamingBudget.h"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <system_error>

namespace torch_tensorrt {
namespace executorch_backend {

WsBudget parse_weight_streaming_budget(const void* value, std::size_t nbytes) {
  WsBudget result; // valid == false until the value is fully parsed

  if (value == nullptr || nbytes == 0) {
    return result;
  }
  // The value is a non-negative decimal byte budget and is not NUL terminated.
  // std::from_chars consumes only ASCII digits (no leading whitespace, sign, or
  // base prefix), so a leftover byte (trailing garbage or an embedded NUL), an
  // out-of-range value, or a negative leaves the result invalid.
  const char* const first = static_cast<const char*>(value);
  const char* const last = first + nbytes;
  int64_t parsed = 0;
  const std::from_chars_result fc = std::from_chars(first, last, parsed);
  if (fc.ec != std::errc() || fc.ptr != last || parsed < 0) {
    return result;
  }

  result.valid = true;
  result.bytes = parsed;
  return result;
}

} // namespace executorch_backend
} // namespace torch_tensorrt
