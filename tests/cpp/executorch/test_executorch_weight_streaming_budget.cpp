#include "torch_tensorrt/executorch/WeightStreamingBudget.h"

#include "gtest/gtest.h"

#include <cstddef>
#include <cstring>
#include <limits>

namespace torch_tensorrt {
namespace executorch_backend {
namespace {

// The compile spec value is a raw byte range; parse a C string literal by length.
WsBudget parse(const char* s) {
  return parse_weight_streaming_budget(s, std::strlen(s));
}

TEST(ExecuTorchWeightStreamingBudget, ParsesZeroBytes) {
  const WsBudget budget = parse("0");
  EXPECT_TRUE(budget.valid);
  EXPECT_EQ(budget.bytes, 0);
}

TEST(ExecuTorchWeightStreamingBudget, ParsesLargeByteCount) {
  const WsBudget budget = parse("8589934592");
  EXPECT_TRUE(budget.valid);
  EXPECT_EQ(budget.bytes, 8589934592LL);
}

TEST(ExecuTorchWeightStreamingBudget, RejectsNegative) {
  // The budget is a non-negative byte count; negatives are rejected.
  EXPECT_FALSE(parse("-1").valid);
}

TEST(ExecuTorchWeightStreamingBudget, ParsesInt64Max) {
  const WsBudget budget = parse("9223372036854775807");
  EXPECT_TRUE(budget.valid);
  EXPECT_EQ(budget.bytes, std::numeric_limits<int64_t>::max());
}

TEST(ExecuTorchWeightStreamingBudget, RejectsLargeNegative) {
  EXPECT_FALSE(parse("-9223372036854775808").valid);
}

TEST(ExecuTorchWeightStreamingBudget, RejectsOverflow) {
  // One past the int64_t maximum, so from_chars reports result_out_of_range.
  EXPECT_FALSE(parse("9223372036854775808").valid);
}

TEST(ExecuTorchWeightStreamingBudget, RejectsEmpty) {
  EXPECT_FALSE(parse("").valid);
}

TEST(ExecuTorchWeightStreamingBudget, RejectsGarbage) {
  EXPECT_FALSE(parse("garbage").valid);
}

TEST(ExecuTorchWeightStreamingBudget, RejectsTrailingNonNumeric) {
  EXPECT_FALSE(parse("12x").valid);
}

TEST(ExecuTorchWeightStreamingBudget, RejectsInternalWhitespace) {
  EXPECT_FALSE(parse("12 34").valid);
}

TEST(ExecuTorchWeightStreamingBudget, RejectsTrailingWhitespace) {
  EXPECT_FALSE(parse("42 ").valid);
}

TEST(ExecuTorchWeightStreamingBudget, ParsesNonNulTerminatedBuffer) {
  // The compile spec value is a raw byte range, not a C string. Place the value
  // inside a larger buffer with no terminator after it and parse only its bytes.
  const char raw[] = {'1', '2', '8', 'X', 'Y', 'Z'};
  const WsBudget budget = parse_weight_streaming_budget(raw, 3);
  EXPECT_TRUE(budget.valid);
  EXPECT_EQ(budget.bytes, 128);
}

TEST(ExecuTorchWeightStreamingBudget, ZeroLengthBufferIsMalformed) {
  EXPECT_FALSE(parse_weight_streaming_budget("0", 0).valid);
}

TEST(ExecuTorchWeightStreamingBudget, ParsesLeadingZeros) {
  // Leading zeros are accepted as long as the value fits in int64.
  const WsBudget budget = parse("0000000001");
  EXPECT_TRUE(budget.valid);
  EXPECT_EQ(budget.bytes, 1);
}

TEST(ExecuTorchWeightStreamingBudget, RejectsEmbeddedNul) {
  // The compile spec value is a raw byte range, so an interior NUL must not let
  // the value parse as just the bytes before it.
  const char raw[] = {'1', '2', '3', '\0', 'x'};
  EXPECT_FALSE(parse_weight_streaming_budget(raw, sizeof(raw)).valid);
}

TEST(ExecuTorchWeightStreamingBudget, RejectsLeadingWhitespace) {
  // from_chars does not skip leading whitespace, so " 42" is rejected.
  EXPECT_FALSE(parse(" 42").valid);
}

TEST(ExecuTorchWeightStreamingBudget, RejectsLeadingPlus) {
  // from_chars does not accept a leading plus sign.
  EXPECT_FALSE(parse("+42").valid);
}

} // namespace
} // namespace executorch_backend
} // namespace torch_tensorrt
