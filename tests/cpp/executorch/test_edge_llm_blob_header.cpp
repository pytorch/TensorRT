#include "torch_tensorrt/executorch/EdgeLLMBlobHeader.h"

#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace torch_tensorrt {
namespace executorch_backend {
namespace {

constexpr char EDGE_LLM_MAGIC[4] = {'E', 'L', '0', '1'};
constexpr uint32_t HEADER_SIZE = 32;
constexpr uint32_t BLOB_ALIGNMENT = 16;

template <typename T>
void write_field(std::vector<uint8_t>& payload, std::size_t offset, T value) {
  std::memcpy(payload.data() + offset, &value, sizeof(value));
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

std::string valid_metadata() {
  return R"({"abi_version":1,"component":"vision","outputs":[{"dtype":"float16","shape":[1,4,8]}],)"
         R"("runner":"vit","runner_config":{"model_type":"vit"}})";
}

std::vector<uint8_t> make_payload(const std::string& metadata, const std::string& nested = "TR01nested") {
  const auto metadata_offset = static_cast<uint32_t>(HEADER_SIZE);
  const auto metadata_size = static_cast<uint32_t>(metadata.size());
  const auto blob_offset = static_cast<uint32_t>(align_up(metadata_offset + metadata_size, BLOB_ALIGNMENT));
  std::vector<uint8_t> payload(blob_offset + nested.size(), 0);
  std::memcpy(payload.data(), EDGE_LLM_MAGIC, sizeof(EDGE_LLM_MAGIC));
  write_field(payload, 4, metadata_offset);
  write_field(payload, 8, metadata_size);
  write_field(payload, 12, blob_offset);
  write_field(payload, 16, static_cast<uint64_t>(nested.size()));
  std::memcpy(payload.data() + metadata_offset, metadata.data(), metadata.size());
  std::memcpy(payload.data() + blob_offset, nested.data(), nested.size());
  return payload;
}

TEST(ExecuTorchEdgeLLMBlobHeader, ParsesVisionPayload) {
  const auto payload = make_payload(valid_metadata());

  EdgeLLMBlobHeader header;
  ASSERT_TRUE(EdgeLLMBlobHeader::parse(payload.data(), payload.size(), header));
  EXPECT_EQ(header.abi_version, 1);
  EXPECT_EQ(header.component, "vision");
  EXPECT_EQ(header.runner, "vit");
  EXPECT_EQ(header.runner_config_json, R"({"model_type":"vit"})");
  EXPECT_EQ(header.blob_offset % BLOB_ALIGNMENT, 0);
  EXPECT_EQ(EdgeLLMBlobHeader::nested_blob_data(payload.data(), header), payload.data() + header.blob_offset);
}

TEST(ExecuTorchEdgeLLMBlobHeader, RejectsInvalidMagic) {
  auto payload = make_payload(valid_metadata());
  payload[0] = 'X';
  EdgeLLMBlobHeader header;
  EXPECT_FALSE(EdgeLLMBlobHeader::parse(payload.data(), payload.size(), header));
}

TEST(ExecuTorchEdgeLLMBlobHeader, RejectsUnsupportedAbi) {
  auto metadata = valid_metadata();
  metadata.replace(metadata.find("\"abi_version\":1"), 15, "\"abi_version\":2");
  const auto payload = make_payload(metadata);
  EdgeLLMBlobHeader header;
  EXPECT_FALSE(EdgeLLMBlobHeader::parse(payload.data(), payload.size(), header));
}

TEST(ExecuTorchEdgeLLMBlobHeader, RejectsWrongRunner) {
  auto metadata = valid_metadata();
  metadata.replace(metadata.find("\"runner\":\"vit\""), 14, "\"runner\":\"qwen\"");
  const auto payload = make_payload(metadata);
  EdgeLLMBlobHeader header;
  EXPECT_FALSE(EdgeLLMBlobHeader::parse(payload.data(), payload.size(), header));
}

TEST(ExecuTorchEdgeLLMBlobHeader, RejectsEmptyOutputs) {
  auto metadata = valid_metadata();
  const auto outputs = metadata.find("\"outputs\":[");
  const auto outputs_end = metadata.find(']', outputs);
  metadata.replace(outputs + 10, outputs_end - outputs - 9, "[]");
  const auto payload = make_payload(metadata);
  EdgeLLMBlobHeader header;
  EXPECT_FALSE(EdgeLLMBlobHeader::parse(payload.data(), payload.size(), header));
}

TEST(ExecuTorchEdgeLLMBlobHeader, RejectsNestedBlobPastPayload) {
  auto payload = make_payload(valid_metadata());
  write_field(payload, 16, static_cast<uint64_t>(payload.size()));
  EdgeLLMBlobHeader header;
  EXPECT_FALSE(EdgeLLMBlobHeader::parse(payload.data(), payload.size(), header));
}

} // namespace
} // namespace executorch_backend
} // namespace torch_tensorrt
