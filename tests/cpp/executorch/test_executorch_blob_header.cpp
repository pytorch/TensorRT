#include "torch_tensorrt/executorch/TensorRTBlobHeader.h"

#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace torch_tensorrt {
namespace executorch_backend {
namespace {

constexpr char TENSORRT_MAGIC[4] = {'T', 'R', '0', '1'};
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

std::vector<uint8_t> make_blob(const std::string& metadata, std::size_t engine_size = 4) {
  const auto metadata_offset = static_cast<uint32_t>(HEADER_SIZE);
  const auto metadata_size = static_cast<uint32_t>(metadata.size());
  const auto engine_offset = static_cast<uint32_t>(align_up(metadata_offset + metadata_size, ENGINE_ALIGNMENT));
  std::vector<uint8_t> blob(static_cast<std::size_t>(engine_offset) + engine_size, 0);

  std::memcpy(blob.data(), TENSORRT_MAGIC, sizeof(TENSORRT_MAGIC));
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

} // namespace
} // namespace executorch_backend
} // namespace torch_tensorrt
