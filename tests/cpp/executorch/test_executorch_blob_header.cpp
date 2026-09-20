/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "torch_tensorrt/executorch/TensorRTBlobHeader.h"

#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace torch_tensorrt {
namespace executorch_backend {
namespace {

constexpr char TENSORRT_MAGIC[4] = {'T', 'R', '0', '1'};
constexpr char TENSORRT_MAGIC_ALIASED_IO[4] = {'T', 'R', '0', '2'};
constexpr uint32_t METADATA_OFFSET_FIELD_OFFSET = 4;
constexpr uint32_t METADATA_SIZE_FIELD_OFFSET = 8;
constexpr uint32_t ENGINE_OFFSET_FIELD_OFFSET = 12;
constexpr uint32_t ENGINE_SIZE_FIELD_OFFSET = 16;
constexpr uint32_t HEADER_SIZE = 32;
constexpr uint32_t ENGINE_ALIGNMENT = 16;

template <typename T>
void write_field(std::vector<uint8_t>& blob, std::size_t offset, T value) {
  std::memcpy(blob.data() + offset, &value, sizeof(value));
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

std::vector<uint8_t> make_blob(
    const std::string& metadata,
    std::size_t engine_size = 4,
    const char* magic = TENSORRT_MAGIC) {
  const auto metadata_offset = static_cast<uint32_t>(HEADER_SIZE);
  const auto metadata_size = static_cast<uint32_t>(metadata.size());
  const auto engine_offset = static_cast<uint32_t>(align_up(metadata_offset + metadata_size, ENGINE_ALIGNMENT));
  std::vector<uint8_t> blob(static_cast<std::size_t>(engine_offset) + engine_size, 0);

  std::memcpy(blob.data(), magic, sizeof(TENSORRT_MAGIC));
  write_field(blob, METADATA_OFFSET_FIELD_OFFSET, metadata_offset);
  write_field(blob, METADATA_SIZE_FIELD_OFFSET, metadata_size);
  write_field(blob, ENGINE_OFFSET_FIELD_OFFSET, engine_offset);
  write_field(blob, ENGINE_SIZE_FIELD_OFFSET, static_cast<uint64_t>(engine_size));
  std::memcpy(blob.data() + metadata_offset, metadata.data(), metadata.size());
  return blob;
}

TEST(ExecuTorchTensorRTBlobHeader, ParsesValidHeaderAndMetadata) {
  const std::string metadata =
      R"({"io_bindings":[{"name":"input_0","is_input":true},{"name":"output_0","is_input":false}],"hardware_compatible":true,"device_id":2})";
  const auto blob = make_blob(metadata);

  TensorRTBlobHeader header;
  ASSERT_TRUE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));

  EXPECT_EQ(header.metadata_offset, HEADER_SIZE);
  EXPECT_EQ(header.metadata_size, metadata.size());
  EXPECT_EQ(header.engine_offset % ENGINE_ALIGNMENT, 0);
  EXPECT_EQ(header.engine_size, 4);
  ASSERT_EQ(header.input_binding_names.size(), 1);
  EXPECT_EQ(header.input_binding_names[0], "input_0");
  ASSERT_EQ(header.output_binding_names.size(), 1);
  EXPECT_EQ(header.output_binding_names[0], "output_0");
  EXPECT_TRUE(header.hardware_compatible);
  EXPECT_EQ(header.device_id, 2);
  EXPECT_EQ(TensorRTBlobHeader::engine_data(blob.data(), header), blob.data() + header.engine_offset);
}

TEST(ExecuTorchTensorRTBlobHeader, RejectsASizeThatWouldWrapWhenAddedToItsOffset) {
  // These sizes come from the file, so a hostile or truncated one can be large enough that adding
  // it to its offset wraps past zero. A check written as offset plus size then reads as small and
  // lets the parse through, after which the reader walks far past the end of the blob. The check
  // has to compare against the space that is left instead, and this is the case that tells the two
  // forms apart: every other input in this file is accepted or rejected identically by both.
  // Metadata that parses, so the only thing left that can refuse the blob is the length check. With
  // "{}" the parse fails for want of a bindings list and the test passes either way, which is how
  // this case originally proved nothing.
  static constexpr const char* kValidMetadata =
      R"({"io_bindings":[{"name":"input_0","is_input":true},{"name":"output_0","is_input":false}]})";
  const std::vector<uint64_t> wrapping = {
      UINT64_MAX,
      UINT64_MAX - 15,
      UINT64_MAX - HEADER_SIZE,
  };
  for (const uint64_t engine_size : wrapping) {
    std::vector<uint8_t> blob = make_blob(kValidMetadata);
    write_field(blob, ENGINE_SIZE_FIELD_OFFSET, engine_size);
    TensorRTBlobHeader header;
    EXPECT_FALSE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header))
        << "engine_size " << engine_size << " must be refused on a blob of " << blob.size() << " bytes";
  }
  // The metadata length is read the same way. This one does not tell the two forms apart, because a
  // 32 bit length cannot wrap a 64 bit sum, but it is the boundary worth pinning anyway.
  std::vector<uint8_t> blob = make_blob(kValidMetadata);
  write_field(blob, METADATA_SIZE_FIELD_OFFSET, static_cast<uint32_t>(UINT32_MAX));
  TensorRTBlobHeader header;
  EXPECT_FALSE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
}

TEST(ExecuTorchTensorRTBlobHeader, RejectsInvalidMagic) {
  auto blob = make_blob(R"({"io_bindings":[]})");
  blob[0] = 'X';

  TensorRTBlobHeader header;
  EXPECT_FALSE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
}

TEST(ExecuTorchTensorRTBlobHeader, RejectsUnalignedEngineOffset) {
  const std::string metadata = "{}";
  const auto metadata_offset = static_cast<uint32_t>(HEADER_SIZE);
  const auto metadata_size = static_cast<uint32_t>(metadata.size());
  const auto engine_offset = static_cast<uint32_t>(HEADER_SIZE + metadata.size());
  ASSERT_NE(engine_offset % ENGINE_ALIGNMENT, 0);

  std::vector<uint8_t> blob(static_cast<std::size_t>(engine_offset) + 4, 0);
  std::memcpy(blob.data(), TENSORRT_MAGIC, sizeof(TENSORRT_MAGIC));
  write_field(blob, METADATA_OFFSET_FIELD_OFFSET, metadata_offset);
  write_field(blob, METADATA_SIZE_FIELD_OFFSET, metadata_size);
  write_field(blob, ENGINE_OFFSET_FIELD_OFFSET, engine_offset);
  write_field(blob, ENGINE_SIZE_FIELD_OFFSET, uint64_t{4});
  std::memcpy(blob.data() + metadata_offset, metadata.data(), metadata.size());

  TensorRTBlobHeader header;
  EXPECT_FALSE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
}

TEST(ExecuTorchTensorRTBlobHeader, RejectsEnginePastEndOfBlob) {
  auto blob = make_blob(R"({"io_bindings":[]})");
  write_field(blob, ENGINE_SIZE_FIELD_OFFSET, static_cast<uint64_t>(blob.size()));

  TensorRTBlobHeader header;
  EXPECT_FALSE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
}

TEST(ExecuTorchTensorRTBlobHeader, RejectsMissingIoBindingsMetadata) {
  const auto blob = make_blob(R"({"hardware_compatible":false})");

  TensorRTBlobHeader header;
  EXPECT_FALSE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
}

TEST(ExecuTorchTensorRTBlobHeader, ParsesAliasedIo) {
  const std::string metadata = R"({"io_bindings":[{"name":"in_k","is_input":true},{"name":"out_k","is_input":false},)"
                               R"({"name":"in_u","is_input":true},{"name":"out_u","is_input":false}],)"
                               R"("aliased_io":[{"output":"out_k","input":"in_k","kind":"kv_cache_update"},)"
                               R"({"output":"out_u","input":"in_u","kind":"user"}],)"
                               R"("hardware_compatible":false,"device_id":0})";
  const auto blob = make_blob(metadata);

  TensorRTBlobHeader header;
  ASSERT_TRUE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));

  ASSERT_EQ(header.aliased_io.size(), 2u);
  EXPECT_EQ(header.aliased_io[0].output, "out_k");
  EXPECT_EQ(header.aliased_io[0].input, "in_k");
  EXPECT_EQ(header.aliased_io[0].kind, "kv_cache_update");
  EXPECT_EQ(header.aliased_io[1].output, "out_u");
  EXPECT_EQ(header.aliased_io[1].input, "in_u");
  EXPECT_EQ(header.aliased_io[1].kind, "user");
}

TEST(ExecuTorchTensorRTBlobHeader, DefaultsMissingAliasedIo) {
  // Blobs written before aliased_io existed omit the key; parsing must still
  // succeed and leave aliased_io empty (backward compatible).
  const auto blob =
      make_blob(R"({"io_bindings":[{"name":"input_0","is_input":true},{"name":"output_0","is_input":false}],)"
                R"("hardware_compatible":false,"device_id":0})");

  TensorRTBlobHeader header;
  ASSERT_TRUE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
  EXPECT_TRUE(header.aliased_io.empty());
}

TEST(ExecuTorchTensorRTBlobHeader, ParsesEmptyAliasedIo) {
  const auto blob =
      make_blob(R"({"io_bindings":[{"name":"input_0","is_input":true},{"name":"output_0","is_input":false}],)"
                R"("aliased_io":[],"hardware_compatible":false,"device_id":0})");

  TensorRTBlobHeader header;
  ASSERT_TRUE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
  EXPECT_TRUE(header.aliased_io.empty());
}

TEST(ExecuTorchTensorRTBlobHeader, ParsesAliasedIoMagic) {
  const std::string metadata = R"({"io_bindings":[{"name":"in_k","is_input":true},{"name":"out_k","is_input":false}],)"
                               R"("aliased_io":[{"output":"out_k","input":"in_k","kind":"kv_cache_update"}]})";
  const auto blob = make_blob(metadata, 4, TENSORRT_MAGIC_ALIASED_IO);

  TensorRTBlobHeader header;
  ASSERT_TRUE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
  ASSERT_EQ(header.aliased_io.size(), 1u);
  EXPECT_EQ(header.aliased_io[0].output, "out_k");
  EXPECT_EQ(header.aliased_io[0].input, "in_k");
}

TEST(ExecuTorchTensorRTBlobHeader, InputNamedAliasedIoWithNoAliasesStillParses) {
  // A model input literally named "aliased_io" must not be mistaken for the
  // real aliased_io array key.
  const auto blob =
      make_blob(R"({"io_bindings":[{"name":"aliased_io","is_input":true},{"name":"out_0","is_input":false}],)"
                R"("hardware_compatible":false,"device_id":0})");

  TensorRTBlobHeader header;
  ASSERT_TRUE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
  EXPECT_TRUE(header.aliased_io.empty());
}

TEST(ExecuTorchTensorRTBlobHeader, MetadataKeyOrderDoesNotChangeWhatIsRead) {
  // JSON does not order keys and sorting them is one word in any writer, so a reader that depends
  // on the order our writer happens to emit loses aliases silently. For a KV cache program that is
  // wrong answers rather than a failure, because every in-place update lands in the delegate's own
  // output slot instead of the caller's buffer.
  const std::string bindings = R"("io_bindings":[{"name":"in_k","is_input":true},{"name":"out_k","is_input":false}])";
  const std::string aliases = R"("aliased_io":[{"output":"out_k","input":"in_k","kind":"kv_cache_update"}])";
  const std::string scalars = R"("device_id":3,"hardware_compatible":true)";

  for (const std::string& metadata :
       {"{" + bindings + "," + aliases + "," + scalars + "}",
        "{" + aliases + "," + scalars + "," + bindings + "}",
        "{" + scalars + "," + bindings + "," + aliases + "}"}) {
    const auto blob = make_blob(metadata, 4, TENSORRT_MAGIC_ALIASED_IO);

    TensorRTBlobHeader header;
    ASSERT_TRUE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header)) << metadata;
    ASSERT_EQ(header.aliased_io.size(), 1u) << metadata;
    EXPECT_EQ(header.aliased_io[0].output, "out_k") << metadata;
    EXPECT_EQ(header.aliased_io[0].input, "in_k") << metadata;
    EXPECT_EQ(header.device_id, 3) << metadata;
    EXPECT_TRUE(header.hardware_compatible) << metadata;
  }
}

TEST(ExecuTorchTensorRTBlobHeader, InputNamedLikeAScalarKeyIsNotReadAsOne) {
  // The io_bindings array holds caller-chosen tensor names, which is why the scalars are not simply
  // searched for across the whole object.
  const auto blob = make_blob(
      R"({"io_bindings":[{"name":"device_id","is_input":true},{"name":"out_0","is_input":false}],)"
      R"("device_id":3,"hardware_compatible":true})",
      4,
      TENSORRT_MAGIC_ALIASED_IO);

  TensorRTBlobHeader header;
  ASSERT_TRUE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
  EXPECT_EQ(header.device_id, 3);
  EXPECT_TRUE(header.hardware_compatible);
}

TEST(ExecuTorchTensorRTBlobHeader, RejectsADeviceIdThatDoesNotFitAnInt) {
  // The device id goes to cudaSetDevice, so a value the reader cannot hold has to stop the parse.
  // Wrapping it instead yields a small plausible number: 4294967299 came back as device 3, which
  // runs the engine on the wrong card on a machine that has one.
  for (const char* too_large : {"2147483648", "4294967299", "99999999999999999999", "-2147483649"}) {
    const auto blob =
        make_blob(R"({"io_bindings":[{"name":"x","is_input":true}],"device_id":)" + std::string(too_large) + "}");

    TensorRTBlobHeader header;
    EXPECT_FALSE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header)) << "device_id " << too_large;
  }
}

TEST(ExecuTorchTensorRTBlobHeader, ParsesTheEndsOfTheDeviceIdRange) {
  for (const int expected : {0, 7, -1, std::numeric_limits<int>::max(), std::numeric_limits<int>::min()}) {
    const auto blob =
        make_blob(R"({"io_bindings":[{"name":"x","is_input":true}],"device_id":)" + std::to_string(expected) + "}");

    TensorRTBlobHeader header;
    ASSERT_TRUE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header)) << "device_id " << expected;
    EXPECT_EQ(header.device_id, expected);
  }
}

TEST(ExecuTorchTensorRTBlobHeader, RejectsUnknownFutureMagic) {
  constexpr char kFutureMagic[4] = {'T', 'R', '0', '3'};
  const std::string metadata = R"({"io_bindings":[{"name":"x","is_input":true}]})";
  const auto blob = make_blob(metadata, 4, kFutureMagic);

  TensorRTBlobHeader header;
  EXPECT_FALSE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
}

} // namespace
} // namespace executorch_backend
} // namespace torch_tensorrt
