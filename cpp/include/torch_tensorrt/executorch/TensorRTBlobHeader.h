/**
 * @file TensorRTBlobHeader.h
 * @brief The TR01 processed-blob layout the ExecuTorch delegate is handed.
 *
 * Written by py/torch_tensorrt/executorch/serialization.py: a 32-byte fixed
 * header carrying the magic `TR01` and four offset/size fields, then JSON
 * metadata, then the serialized TensorRT engine. Deliberately standalone, so
 * loading a `.pte` needs neither the legacy Torch-TensorRT C++ runtime nor
 * libtorch.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace torch_tensorrt {
namespace executorch_backend {

// One aliased output->input binding pair (KV-cache in-place update, or a
// user-declared alias). The engine's output binding shares device memory with
// the named input binding; the runtime binds the output to the input's tensor
// so the update lands in-place in the caller-owned buffer.
struct AliasedBinding {
  std::string output; // output binding name
  std::string input; // input binding name it aliases
  std::string kind; // "kv_cache_update" (TRT-enforced) or "user"
};

/// @brief Where the metadata and engine live inside a TR01 blob, and what the metadata said.
struct TensorRTBlobHeader {
  uint32_t metadata_offset = 0; ///< Byte offset of the JSON metadata; never inside the header.
  uint32_t metadata_size = 0; ///< Length of the JSON metadata in bytes.
  uint32_t engine_offset = 0; ///< Byte offset of the serialized engine; 16-byte aligned.
  uint64_t engine_size = 0; ///< Length of the serialized engine in bytes.
  /// @brief TensorRT input binding names, in the order the metadata lists them.
  std::vector<std::string> input_binding_names;
  /// @brief TensorRT output binding names, in the order the metadata lists them.
  std::vector<std::string> output_binding_names;
  /// @brief Aliased output->input binding pairs declared by the metadata.
  std::vector<AliasedBinding> aliased_io;
  bool hardware_compatible = false; ///< Engine built hardware-compatible; false when unstated.
  int device_id = 0; ///< Device the engine was built for; 0 when unstated.

  /**
   * @brief Start of the serialized engine inside a blob.
   *
   * @param blob Start of the processed blob `h` was parsed from.
   * @param h Header parsed from that blob.
   * @return Pointer to TensorRTBlobHeader::engine_size bytes of engine data.
   */
  static const void* engine_data(const void* blob, const TensorRTBlobHeader& h);

  /**
   * @brief Validates a processed blob and reads its header.
   *
   * Checks the `TR01` magic, that the metadata and the engine both lie within
   * `size`, that the engine is 16-byte aligned, and that the metadata ends at or
   * before the engine; then reads the binding names and flags out of the JSON.
   *
   * @param data Start of the processed blob.
   * @param size Bytes available at `data`.
   * @param[out] out Filled in on success; left in an unspecified state otherwise.
   * @return false when the blob is not a well-formed TR01 blob.
   */
  static bool parse(const void* data, std::size_t size, TensorRTBlobHeader& out);
};

} // namespace executorch_backend
} // namespace torch_tensorrt
