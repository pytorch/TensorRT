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

// Metadata a valid blob would carry, so a case that corrupts one header field is refused by the
// check it targets and not by the metadata parser.
constexpr const char* VALID_METADATA =
    R"({"io_bindings":[{"name":"input_0","is_input":true},{"name":"output_0","is_input":false}]})";

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

TEST(ExecuTorchTensorRTBlobHeader, ReadsTheDeviceTheProgramAsksForWhateverElseIsInTheMetadata) {
  // A real program asking for device 9 was read as asking for device 0. The reader searched the whole
  // document for the key and excluded the ranges the arrays occupied, but those ranges are computed
  // while walking, so one unrelated key shifted what they covered and the search matched inside it.
  auto blob = make_blob(R"({"io_bindings":[{"name":"x","is_input":true}],"unrelated":{"device_id":0},)"
                        R"("device_id":9,"hardware_compatible":false})");
  TensorRTBlobHeader header;
  ASSERT_TRUE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
  EXPECT_EQ(header.device_id, 9);

  // A tensor may legitimately carry the same name as a key, and the key is still the object's own.
  auto named = make_blob(R"({"io_bindings":[{"name":"device_id","is_input":true}],"device_id":7,)"
                         R"("hardware_compatible":false})");
  TensorRTBlobHeader named_header;
  ASSERT_TRUE(TensorRTBlobHeader::parse(named.data(), named.size(), named_header));
  EXPECT_EQ(named_header.device_id, 7);
}

TEST(ExecuTorchTensorRTBlobHeader, RejectsInvalidMagic) {
  auto blob = make_blob(R"({"io_bindings":[]})");
  blob[0] = 'X';

  TensorRTBlobHeader header;
  EXPECT_FALSE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
}

TEST(ExecuTorchTensorRTBlobHeader, RejectsUnalignedEngineOffset) {
  const std::string metadata = R"({"io_bindings":[{"name":"input_0","is_input":true}]})";
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

// A program file is untrusted input, so each bounds check below gets a case of its own. Without one
// the fields are read out of a buffer that does not hold them.
TEST(ExecuTorchTensorRTBlobHeader, RejectsABlobShorterThanTheHeader) {
  // Refusing has to happen before the fields are read, and the return value alone cannot show
  // that: a four byte buffer reads its metadata offset out of memory it does not own, gets a
  // small number, and is then refused by the offset floor instead. So the length check could be
  // deleted outright with every case in this file still passing. What the length check alone
  // decides is whether anything past the caller's size is read at all, so the buffer here holds
  // a header that parses and only the size says it is not there. A reader that trusts the bytes
  // instead of the size fills the output in from them, which is what the last assertion catches.
  const auto valid = make_blob(VALID_METADATA);
  for (const std::size_t short_size : {std::size_t{0}, sizeof(TENSORRT_MAGIC), std::size_t{HEADER_SIZE} - 1}) {
    std::vector<uint8_t> blob(valid);
    TensorRTBlobHeader header;
    EXPECT_FALSE(TensorRTBlobHeader::parse(blob.data(), short_size, header))
        << "a blob of " << short_size << " bytes is shorter than the " << HEADER_SIZE << " byte header";
    EXPECT_EQ(header.metadata_offset, 0u) << "a field was read past the " << short_size << " bytes the caller offered";
    EXPECT_EQ(header.metadata_size, 0u);
    EXPECT_EQ(header.engine_offset, 0u);
    EXPECT_EQ(header.engine_size, 0u);
  }
}

TEST(ExecuTorchTensorRTBlobHeader, RejectsAnEngineOffsetPastEndOfBlob) {
  auto blob = make_blob(VALID_METADATA);
  const auto past_end = static_cast<uint32_t>(align_up(blob.size() + ENGINE_ALIGNMENT, ENGINE_ALIGNMENT));
  write_field(blob, ENGINE_OFFSET_FIELD_OFFSET, past_end);
  write_field(blob, ENGINE_SIZE_FIELD_OFFSET, uint64_t{0});

  TensorRTBlobHeader header;
  EXPECT_FALSE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
}

TEST(ExecuTorchTensorRTBlobHeader, RejectsMetadataThatRunsIntoTheEngine) {
  auto blob = make_blob(VALID_METADATA);
  uint32_t engine_offset = 0;
  std::memcpy(&engine_offset, blob.data() + ENGINE_OFFSET_FIELD_OFFSET, sizeof(engine_offset));
  write_field(blob, METADATA_SIZE_FIELD_OFFSET, static_cast<uint32_t>(engine_offset - HEADER_SIZE + 1));

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
  // searched for across the whole object. Written without the aliased-io magic, because that magic
  // now promises an alias array and this blob carries none.
  const auto blob = make_blob(
      R"({"io_bindings":[{"name":"device_id","is_input":true},{"name":"out_0","is_input":false}],)"
      R"("device_id":3,"hardware_compatible":true})");

  TensorRTBlobHeader header;
  ASSERT_TRUE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
  EXPECT_EQ(header.device_id, 3);
  EXPECT_TRUE(header.hardware_compatible);
}

TEST(ExecuTorchTensorRTBlobHeader, RejectsRepeatedAliasedIoOutput) {
  // Two entries claiming out_k. They resolve to the same binding and the same
  // engine alias, so nothing that looks the names up can tell them from one
  // entry -- but a reader counting aliased outputs per entry counts out_k twice,
  // which is how the backend sizes the delegate argument list.
  const std::string metadata = R"({"io_bindings":[{"name":"in_k","is_input":true},{"name":"in_v","is_input":true},)"
                               R"({"name":"out_k","is_input":false},{"name":"out_v","is_input":false}],)"
                               R"("aliased_io":[{"output":"out_k","input":"in_k","kind":"kv_cache_update"},)"
                               R"({"output":"out_k","input":"in_k","kind":"kv_cache_update"}]})";
  const auto blob = make_blob(metadata, 4, TENSORRT_MAGIC_ALIASED_IO);

  TensorRTBlobHeader header;
  EXPECT_FALSE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
}

TEST(ExecuTorchTensorRTBlobHeader, RejectsRepeatedAliasedIoOutputWithADifferentInput) {
  // The same repeat with the second entry naming a different input. This is the
  // shape the check above exists for: keying the refusal on the output/input
  // pair instead of the output alone would accept it, and out_k's aliased-output
  // count would be two for one output binding, which is the arity the backend
  // subtracts on.
  const std::string metadata = R"({"io_bindings":[{"name":"in_k","is_input":true},{"name":"in_v","is_input":true},)"
                               R"({"name":"out_k","is_input":false},{"name":"out_v","is_input":false}],)"
                               R"("aliased_io":[{"output":"out_k","input":"in_k","kind":"kv_cache_update"},)"
                               R"({"output":"out_k","input":"in_v","kind":"kv_cache_update"}]})";
  const auto blob = make_blob(metadata, 4, TENSORRT_MAGIC_ALIASED_IO);

  TensorRTBlobHeader header;
  EXPECT_FALSE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
}

TEST(ExecuTorchTensorRTBlobHeader, RejectsRepeatedOutputBindingName) {
  // One name, two output slots. Every name lookup stops at the first slot, so
  // init would record the alias there, and execute() would then bind the second
  // slot's ExecuTorch storage to the same TensorRT name -- replacing the address
  // of the caller's buffer that the alias exists to write.
  const std::string metadata = R"({"io_bindings":[{"name":"in_k","is_input":true},)"
                               R"({"name":"out_k","is_input":false},{"name":"out_k","is_input":false}],)"
                               R"("aliased_io":[{"output":"out_k","input":"in_k","kind":"kv_cache_update"}]})";
  const auto blob = make_blob(metadata, 4, TENSORRT_MAGIC_ALIASED_IO);

  TensorRTBlobHeader header;
  EXPECT_FALSE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
}

TEST(ExecuTorchTensorRTBlobHeader, RejectsRepeatedInputBindingName) {
  const std::string metadata = R"({"io_bindings":[{"name":"in_k","is_input":true},{"name":"in_k","is_input":true},)"
                               R"({"name":"out_k","is_input":false}]})";
  const auto blob = make_blob(metadata);

  TensorRTBlobHeader header;
  EXPECT_FALSE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
}

TEST(ExecuTorchTensorRTBlobHeader, RejectsOneNameUsedAsBothAnInputAndAnOutput) {
  // TensorRT has one name space for its tensors, so this is the same collision
  // as the two above rather than a distinct input and output that happen to
  // share a spelling.
  const std::string metadata = R"({"io_bindings":[{"name":"kv","is_input":true},{"name":"kv","is_input":false}]})";
  const auto blob = make_blob(metadata);

  TensorRTBlobHeader header;
  EXPECT_FALSE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
}

TEST(ExecuTorchTensorRTBlobHeader, ParsesAliasedIoEntriesForDistinctOutputs) {
  // The minimal pair for the test above: the same blob with the second entry
  // claiming its own output. A second entry is not itself the defect.
  const std::string metadata = R"({"io_bindings":[{"name":"in_k","is_input":true},{"name":"in_v","is_input":true},)"
                               R"({"name":"out_k","is_input":false},{"name":"out_v","is_input":false}],)"
                               R"("aliased_io":[{"output":"out_k","input":"in_k","kind":"kv_cache_update"},)"
                               R"({"output":"out_v","input":"in_v","kind":"kv_cache_update"}]})";
  const auto blob = make_blob(metadata, 4, TENSORRT_MAGIC_ALIASED_IO);

  TensorRTBlobHeader header;
  ASSERT_TRUE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
  ASSERT_EQ(header.aliased_io.size(), 2u);
  EXPECT_EQ(header.aliased_io[0].output, "out_k");
  EXPECT_EQ(header.aliased_io[1].output, "out_v");
}

TEST(ExecuTorchTensorRTBlobHeader, RejectsRepeatedAliasedIoInputForDifferentOutputs) {
  // Two entries naming different outputs and one input. Both resolve to the same
  // input index, so execute() binds both output bindings to that input's caller
  // pointer -- one address with two writers, the second of which erases the
  // first with no error. The kind here is "user", which init validates only by
  // comparing shapes, so two same-shaped outputs get past it.
  const std::string metadata = R"({"io_bindings":[{"name":"in_0","is_input":true},)"
                               R"({"name":"out_0","is_input":false},{"name":"out_1","is_input":false}],)"
                               R"("aliased_io":[{"output":"out_0","input":"in_0","kind":"user"},)"
                               R"({"output":"out_1","input":"in_0","kind":"user"}]})";
  const auto blob = make_blob(metadata, 4, TENSORRT_MAGIC_ALIASED_IO);

  TensorRTBlobHeader header;
  EXPECT_FALSE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
}

TEST(ExecuTorchTensorRTBlobHeader, RejectsEmptyBindingName) {
  // Skipping a blank name instead of refusing it shortens the recorded output
  // list while the delegate's argument list keeps its length: here one aliased
  // output would be recorded and one real output dropped, so the aliased
  // binding would consume the argument belonging to the dropped one.
  const std::string metadata = R"({"io_bindings":[{"name":"in_k","is_input":true},)"
                               R"({"name":"out_k","is_input":false},{"name":"","is_input":false}],)"
                               R"("aliased_io":[{"output":"out_k","input":"in_k","kind":"kv_cache_update"}]})";
  const auto blob = make_blob(metadata, 4, TENSORRT_MAGIC_ALIASED_IO);

  TensorRTBlobHeader header;
  EXPECT_FALSE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
}

TEST(ExecuTorchTensorRTBlobHeader, RejectsAliasedIoEntryWithABlankInput) {
  // Skipping this entry rather than refusing it records one alias for the two
  // the engine has, and execute() subtracts the recorded count from the delegate
  // argument list, so the .pte fails its arity check on every call with a
  // message that never mentions aliasing.
  const std::string metadata = R"({"io_bindings":[{"name":"in_k","is_input":true},{"name":"in_v","is_input":true},)"
                               R"({"name":"out_k","is_input":false},{"name":"out_v","is_input":false}],)"
                               R"("aliased_io":[{"output":"out_k","input":"in_k","kind":"kv_cache_update"},)"
                               R"({"output":"out_v","input":"","kind":"kv_cache_update"}]})";
  const auto blob = make_blob(metadata, 4, TENSORRT_MAGIC_ALIASED_IO);

  TensorRTBlobHeader header;
  EXPECT_FALSE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
}

TEST(ExecuTorchTensorRTBlobHeader, RejectsAliasedIoEntryWithNoOutputKey) {
  const std::string metadata = R"({"io_bindings":[{"name":"in_k","is_input":true},)"
                               R"({"name":"out_k","is_input":false}],)"
                               R"("aliased_io":[{"input":"in_k","kind":"kv_cache_update"}]})";
  const auto blob = make_blob(metadata, 4, TENSORRT_MAGIC_ALIASED_IO);

  TensorRTBlobHeader header;
  EXPECT_FALSE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
}

TEST(ExecuTorchTensorRTBlobHeader, RejectsBindingEntryWithNoNameKey) {
  const std::string metadata = R"({"io_bindings":[{"name":"in_k","is_input":true},{"is_input":false}]})";
  const auto blob = make_blob(metadata);

  TensorRTBlobHeader header;
  EXPECT_FALSE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
}

TEST(ExecuTorchTensorRTBlobHeader, RejectsWrappingEngineExtent) {
  // engine_size is a 64-bit field read straight from the file, so adding it to
  // engine_offset before comparing against the blob length wraps: 4096 plus
  // 2^64-4086 is 10, which is comfortably inside an 8 KiB blob. The pointer and
  // that length are what TensorRTBackend hands deserializeCudaEngine.
  const std::string metadata = R"({"io_bindings":[{"name":"x","is_input":true}]})";
  constexpr std::size_t kBlobSize = 8192;
  constexpr uint32_t kMetadataOffset = HEADER_SIZE;
  constexpr uint32_t kEngineOffset = 4096;

  std::vector<uint8_t> blob(kBlobSize, 0);
  std::memcpy(blob.data(), TENSORRT_MAGIC, sizeof(TENSORRT_MAGIC));
  write_field(blob, METADATA_OFFSET_FIELD_OFFSET, kMetadataOffset);
  write_field(blob, METADATA_SIZE_FIELD_OFFSET, static_cast<uint32_t>(metadata.size()));
  write_field(blob, ENGINE_OFFSET_FIELD_OFFSET, kEngineOffset);
  write_field(blob, ENGINE_SIZE_FIELD_OFFSET, ~uint64_t{0} - (kEngineOffset - 11));
  std::memcpy(blob.data() + kMetadataOffset, metadata.data(), metadata.size());

  TensorRTBlobHeader header;
  EXPECT_FALSE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
}

TEST(ExecuTorchTensorRTBlobHeader, RejectsEngineOffsetPastEndOfBlob) {
  // The offset alone is out of range. The subtraction form has to refuse that
  // before it evaluates size - engine_offset, which would itself wrap.
  auto blob = make_blob(R"({"io_bindings":[{"name":"x","is_input":true}]})");
  const auto past_end = static_cast<uint32_t>(align_up(blob.size() + ENGINE_ALIGNMENT, ENGINE_ALIGNMENT));
  write_field(blob, ENGINE_OFFSET_FIELD_OFFSET, past_end);
  write_field(blob, ENGINE_SIZE_FIELD_OFFSET, uint64_t{0});

  TensorRTBlobHeader header;
  EXPECT_FALSE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
}

TEST(ExecuTorchTensorRTBlobHeader, RejectsMetadataThatStartsPastTheEngine) {
  // Both metadata fields are in range of the blob, so only the clause comparing
  // them against engine_offset refuses this. That clause is written in the same
  // subtraction form as the engine extent, and the subtraction is what needs the
  // ordering test in front of it: engine_offset - metadata_offset is unsigned,
  // so with the metadata past the engine it wraps to nearly 2^32 and any
  // metadata_size fits under it.
  const std::string metadata = R"({"io_bindings":[]})";
  constexpr std::size_t kBlobSize = 8192;
  constexpr uint32_t kMetadataOffset = 4096;
  constexpr uint32_t kEngineOffset = 48;

  std::vector<uint8_t> blob(kBlobSize, 0);
  std::memcpy(blob.data(), TENSORRT_MAGIC, sizeof(TENSORRT_MAGIC));
  write_field(blob, METADATA_OFFSET_FIELD_OFFSET, kMetadataOffset);
  write_field(blob, METADATA_SIZE_FIELD_OFFSET, static_cast<uint32_t>(metadata.size()));
  write_field(blob, ENGINE_OFFSET_FIELD_OFFSET, kEngineOffset);
  write_field(blob, ENGINE_SIZE_FIELD_OFFSET, uint64_t{4});
  std::memcpy(blob.data() + kMetadataOffset, metadata.data(), metadata.size());

  TensorRTBlobHeader header;
  EXPECT_FALSE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
}

TEST(ExecuTorchTensorRTBlobHeader, RejectsBindingEntryWithNoIsInputKey) {
  // Without the key the initializer in the parser reads the binding as an output
  // while serialization.py's TensorRTIOBinding reads it as an input, so the two
  // readers of these bytes disagree -- and the disagreement is not one slot: it
  // moves in_v out of the input list, which shifts every index after it.
  const std::string metadata = R"({"io_bindings":[{"name":"in_k","is_input":true},{"name":"in_v"},)"
                               R"({"name":"out_k","is_input":false}]})";
  const auto blob = make_blob(metadata);

  TensorRTBlobHeader header;
  EXPECT_FALSE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
}

TEST(ExecuTorchTensorRTBlobHeader, RejectsBindingEntryWithAMisspelledIsInputKey) {
  // One byte wrong is the same case: the key falls through to skip_value, which
  // consumes the value and leaves the initializer standing.
  const std::string metadata = R"({"io_bindings":[{"name":"in_k","is_input":true},{"name":"in_v","is_inout":true},)"
                               R"({"name":"out_k","is_input":false}]})";
  const auto blob = make_blob(metadata);

  TensorRTBlobHeader header;
  EXPECT_FALSE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
}

TEST(ExecuTorchTensorRTBlobHeader, ParsesAStringCarryingAnEscape) {
  // parse_string keeps the character after a backslash rather than decoding the
  // escape, which is the right answer for \" \\ and \/ and the wrong one for
  // the rest. Nothing in a blob this backend loads carries an escape at all --
  // binding names come from Edge placeholders torch.export has already reduced
  // to [A-Za-z0-9_], or are output0..N, and the keys and alias kinds are
  // literals in TensorRTBlobMetadata.to_json -- so this pins the scan rather
  // than a refusal: dropping the backslash branch would end the first name at
  // its escaped quote and derail the walk from there.
  const std::string metadata = R"({"io_bindings":[{"name":"a\"b","is_input":true},)"
                               R"({"name":"c\\d","is_input":false},)"
                               R"({"name":"e\/f","is_input":false}]})";
  const auto blob = make_blob(metadata);

  TensorRTBlobHeader header;
  ASSERT_TRUE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
  ASSERT_EQ(header.input_binding_names.size(), 1u);
  EXPECT_EQ(header.input_binding_names[0], "a\"b");
  ASSERT_EQ(header.output_binding_names.size(), 2u);
  EXPECT_EQ(header.output_binding_names[0], "c\\d");
  EXPECT_EQ(header.output_binding_names[1], "e/f");
}

TEST(ExecuTorchTensorRTBlobHeader, RejectsAliasedIoMagicWithNoAliasArrayFound) {
  // TR02 says the metadata carries aliased_io. Here the key is one byte wrong,
  // so the walk finds nothing and the header would come back alias-free -- and
  // an alias-free header of a threaded .pte binds each aliased output to its own
  // storage and stops updating the caller's cache, with nothing to fail on.
  const std::string metadata = R"({"io_bindings":[{"name":"in_k","is_input":true},)"
                               R"({"name":"out_k","is_input":false}],)"
                               R"("aliasedXio":[{"output":"out_k","input":"in_k","kind":"kv_cache_update"}]})";
  const auto blob = make_blob(metadata, 4, TENSORRT_MAGIC_ALIASED_IO);

  TensorRTBlobHeader header;
  EXPECT_FALSE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
}

TEST(ExecuTorchTensorRTBlobHeader, ParsesScalarsPastAnAliasedBindingNamedLikeAKey) {
  // The two scalar scans search the metadata text, and the alias array sits
  // between where io_bindings ends and where those scans used to start, so an
  // aliased binding named device_id was matched as the key: the scan then walked
  // to the next colon, met the kind string, and failed the whole blob.
  const std::string metadata = R"({"io_bindings":[{"name":"device_id","is_input":true},)"
                               R"({"name":"hardware_compatible","is_input":false}],)"
                               R"("aliased_io":[{"output":"hardware_compatible","input":"device_id",)"
                               R"("kind":"kv_cache_update"}],"hardware_compatible":true,"device_id":3})";
  const auto blob = make_blob(metadata, 4, TENSORRT_MAGIC_ALIASED_IO);

  TensorRTBlobHeader header;
  ASSERT_TRUE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
  EXPECT_TRUE(header.hardware_compatible);
  EXPECT_EQ(header.device_id, 3);
  EXPECT_EQ(header.aliased_io.size(), 1u);
}

TEST(ExecuTorchTensorRTBlobHeader, ParsesTheDeviceIdKeyOutsideAnAliasEntryCarryingOne) {
  // The other half of the same window. Being in key position does not tell an
  // unknown key inside an alias entry from the real one, which the alias walk
  // skips and the scans would otherwise read as the field; starting them past
  // the array is what does.
  const std::string metadata = R"({"io_bindings":[{"name":"in_k","is_input":true},)"
                               R"({"name":"out_k","is_input":false}],)"
                               R"("aliased_io":[{"output":"out_k","input":"in_k","kind":"kv_cache_update",)"
                               R"("device_id":9}],"device_id":3})";
  const auto blob = make_blob(metadata, 4, TENSORRT_MAGIC_ALIASED_IO);

  TensorRTBlobHeader header;
  ASSERT_TRUE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
  EXPECT_EQ(header.device_id, 3);
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

TEST(ExecuTorchTensorRTBlobHeader, RejectsADeviceIdPastTheIntMaximum) {
  // Accumulated into an int this wrapped to 1, which is a GPU that exists on
  // most machines: cudaSetDevice then succeeds and the engine deserializes on a
  // device nobody asked for.
  const auto blob = make_blob(R"({"io_bindings":[{"name":"x","is_input":true}],"device_id":4294967297})");

  TensorRTBlobHeader header;
  EXPECT_FALSE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
}

TEST(ExecuTorchTensorRTBlobHeader, ParsesTheLargestDeviceIdAnIntHolds) {
  // The bound is the int maximum itself, not something short of it, so the
  // refusal above is about overflow and not about long-looking values.
  const auto blob = make_blob(R"({"io_bindings":[{"name":"x","is_input":true}],"device_id":2147483647})");

  TensorRTBlobHeader header;
  ASSERT_TRUE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
  EXPECT_EQ(header.device_id, 2147483647);
}

TEST(ExecuTorchTensorRTBlobHeader, ParsesAStringValueThatIsExactlyAScalarKeyName) {
  // A string *value* that reads like the key: quoted the same way, and so a
  // match for the same search. This blob carries no device_id of its own, which
  // is what an older writer emits, so the value is the only match there is --
  // and reading it as the key meant walking to the next colon, meeting the
  // target_platform string, and failing a blob that is perfectly good. What
  // separates the two is that a key is followed by its own colon and a value is
  // followed by a comma.
  const std::string metadata = R"({"io_bindings":[{"name":"x","is_input":true}],)"
                               R"("serialized_metadata":"device_id","target_platform":"linux"})";
  const auto blob = make_blob(metadata);

  TensorRTBlobHeader header;
  ASSERT_TRUE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
  EXPECT_EQ(header.device_id, 0);
}

TEST(ExecuTorchTensorRTBlobHeader, ParsesEveryScalarFromTheWriterKeyOrder) {
  // The key order TensorRTBlobMetadata.to_json emits, with every field present
  // and both scalars set away from their defaults. The scalar scans start past
  // whichever array they last walked, so a field moved ahead of one of them is
  // not found and keeps its C++-side default while the parse still succeeds --
  // this is what fails if that order changes. The Python half of the same rule
  // is test_serialization.py::test_to_json_writes_every_scalar_after_both_arrays.
  const std::string metadata = R"({"io_bindings":[{"name":"in_k","dtype":"float32","shape":[1,2],"is_input":true},)"
                               R"({"name":"out_k","dtype":"float32","shape":[1,2],"is_input":false}],)"
                               R"("aliased_io":[{"output":"out_k","input":"in_k","kind":"kv_cache_update"}],)"
                               R"("hardware_compatible":true,"device_id":6,)"
                               R"("serialized_metadata":"","target_platform":"linux_x86_64"})";
  const auto blob = make_blob(metadata, 4, TENSORRT_MAGIC_ALIASED_IO);

  TensorRTBlobHeader header;
  ASSERT_TRUE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
  EXPECT_EQ(header.input_binding_names, std::vector<std::string>{"in_k"});
  EXPECT_EQ(header.output_binding_names, std::vector<std::string>{"out_k"});
  ASSERT_EQ(header.aliased_io.size(), 1u);
  EXPECT_EQ(header.aliased_io[0].output, "out_k");
  EXPECT_EQ(header.aliased_io[0].input, "in_k");
  EXPECT_EQ(header.aliased_io[0].kind, "kv_cache_update");
  EXPECT_TRUE(header.hardware_compatible);
  EXPECT_EQ(header.device_id, 6);
}

TEST(ExecuTorchTensorRTBlobHeader, ParsesAnArrayKeySpelledByAnEarlierStringValue) {
  // The two array keys are found the way the two scalars are: an occurrence
  // followed by its own colon. Taking the first occurrence anywhere and then
  // the next '[' reads the value below as the key and walks the shape array
  // that follows it, which refuses a blob that is perfectly good.
  const std::string metadata = R"({"serialized_metadata":"io_bindings","shape":[9],)"
                               R"("io_bindings":[{"name":"x","is_input":true}],"device_id":6})";
  const auto blob = make_blob(metadata);

  TensorRTBlobHeader header;
  ASSERT_TRUE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
  EXPECT_EQ(header.input_binding_names, std::vector<std::string>{"x"});
  EXPECT_EQ(header.device_id, 6);
}

TEST(ExecuTorchTensorRTBlobHeader, ParsesTheAliasArrayKeySpelledByAnEarlierStringValue) {
  // The alias key gets the same treatment, and its window is narrower: the
  // search already starts past io_bindings, so only a value between the two
  // arrays can stand in for it -- which is where serialized_metadata sits.
  const std::string metadata = R"({"io_bindings":[{"name":"in_k","is_input":true},)"
                               R"({"name":"out_k","is_input":false}],)"
                               R"("serialized_metadata":"aliased_io","shape":[9],)"
                               R"("aliased_io":[{"output":"out_k","input":"in_k","kind":"kv_cache_update"}],)"
                               R"("device_id":6})";
  const auto blob = make_blob(metadata, 4, TENSORRT_MAGIC_ALIASED_IO);

  TensorRTBlobHeader header;
  ASSERT_TRUE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
  ASSERT_EQ(header.aliased_io.size(), 1u);
  EXPECT_EQ(header.aliased_io[0].output, "out_k");
  EXPECT_EQ(header.device_id, 6);
}

TEST(ExecuTorchTensorRTBlobHeader, RejectsIoBindingsWhoseValueIsNotAnArray) {
  // The array has to be the value of the key, not the next '[' in the text: a
  // blob whose io_bindings is an object is otherwise walked from an unrelated
  // bracket further on, and the entries found there are recorded as this
  // engine's bindings. The array below is shaped like the real one so that
  // walking it succeeds, which is what makes the wrong answer a silent one.
  const std::string metadata = R"({"io_bindings":{"name":"x"},)"
                               R"("elsewhere":[{"name":"y","is_input":true}]})";
  const auto blob = make_blob(metadata);

  TensorRTBlobHeader header;
  EXPECT_FALSE(TensorRTBlobHeader::parse(blob.data(), blob.size(), header));
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
