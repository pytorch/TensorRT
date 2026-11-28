#ifndef TORCH_TENSORRT_RUNTIME_REFIT_HELPERS_H
#define TORCH_TENSORRT_RUNTIME_REFIT_HELPERS_H

#include <cstddef>
#include <cstring>
#include <cmath>
#include <limits>

namespace torch_tensorrt {
namespace runtime {

enum class DataType {
  kFLOAT = 0,
  kHALF = 1,
  kINT8 = 2,
  kINT32 = 3,
  kBOOL = 4
};

inline bool are_weights_equal_exact(const void* a, const void* b, size_t size_in_bytes) {
  return std::memcmp(a, b, size_in_bytes) == 0;
}

inline bool are_floats_equal(float a, float b, float epsilon = 1e-6) {
  float diff = std::fabs(a - b);
  float max_abs = std::fabs(a) > std::fabs(b) ? std::fabs(a) : std::fabs(b);
  return diff <= epsilon * (max_abs > 1.0f ? max_abs : 1.0f);
}

inline bool are_weights_equal_tolerant(const void* a, const void* b, size_t count, DataType dtype, float epsilon = 1e-6) {
  switch (dtype) {
    case DataType::kFLOAT: {
      const float* pa = static_cast<const float*>(a);
      const float* pb = static_cast<const float*>(b);
      for (size_t i = 0; i < count; ++i) {
        if (!are_floats_equal(pa[i], pb[i], epsilon)) {
          return false;
        }
      }
      return true;
    }
    case DataType::kHALF:
    case DataType::kINT8:
    case DataType::kINT32:
    case DataType::kBOOL:
    default: {
      size_t elem_size = 0;
      switch (dtype) {
        case DataType::kHALF: elem_size = 2; break;
        case DataType::kINT8: elem_size = 1; break;
        case DataType::kINT32: elem_size = 4; break;
        case DataType::kBOOL: elem_size = 1; break;
        default: elem_size = 0; break;
      }
      return std::memcmp(a, b, count * elem_size) == 0;
    }
  }
}

inline bool are_weights_robust_equal(const void* a, const void* b, size_t count, DataType dtype, float epsilon = 1e-6) {
  size_t elem_size = 0;
  switch (dtype) {
    case DataType::kFLOAT: elem_size = 4; break;
    case DataType::kHALF: elem_size = 2; break;
    case DataType::kINT8: elem_size = 1; break;
    case DataType::kINT32: elem_size = 4; break;
    case DataType::kBOOL: elem_size = 1; break;
    default: elem_size = 0; break;
  }
  if (are_weights_equal_exact(a, b, count * elem_size)) {
    return true;
  }
  return are_weights_equal_tolerant(a, b, count, dtype, epsilon);
}

} // namespace runtime
} // namespace torch_tensorrt

#endif // TORCH_TENSORRT_RUNTIME_REFIT_HELPERS_H