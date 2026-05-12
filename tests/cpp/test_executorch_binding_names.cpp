#include "torch_tensorrt/executorch/TensorRTBindingNames.h"

#include "gtest/gtest.h"

#include <cstddef>
#include <string>
#include <vector>

namespace torch_tensorrt {
namespace executorch_backend {
namespace detail {
namespace {

TEST(ExecuTorchBindingNames, ParseBindingIndexAcceptsDotAndUnderscoreSuffixes) {
  std::size_t index = 0;

  EXPECT_TRUE(parse_binding_index("input.0", index));
  EXPECT_EQ(index, 0);

  EXPECT_TRUE(parse_binding_index("output_12", index));
  EXPECT_EQ(index, 12);

  EXPECT_TRUE(parse_binding_index("profile.input_3", index));
  EXPECT_EQ(index, 3);
}

TEST(ExecuTorchBindingNames, ParseBindingIndexRejectsMissingOrInvalidSuffixes) {
  std::size_t index = 0;

  EXPECT_FALSE(parse_binding_index("input", index));
  EXPECT_FALSE(parse_binding_index("input.", index));
  EXPECT_FALSE(parse_binding_index("input_", index));
  EXPECT_FALSE(parse_binding_index("input.foo", index));
  EXPECT_FALSE(parse_binding_index("input.3x", index));
}

TEST(ExecuTorchBindingNames, AppendBindingNameOrdersByParsedIndex) {
  std::vector<std::string> names;

  EXPECT_TRUE(append_binding_name(names, "input_1"));
  EXPECT_TRUE(append_binding_name(names, "input_0"));

  ASSERT_EQ(names.size(), 2);
  EXPECT_EQ(names[0], "input_0");
  EXPECT_EQ(names[1], "input_1");
  EXPECT_TRUE(all_binding_names_present(names));
}

TEST(ExecuTorchBindingNames, AppendBindingNameRejectsInvalidDuplicateAndMissingPositions) {
  std::vector<std::string> names;

  EXPECT_FALSE(append_binding_name(names, "input"));

  EXPECT_TRUE(append_binding_name(names, "input_1"));
  EXPECT_FALSE(all_binding_names_present(names));

  EXPECT_TRUE(append_binding_name(names, "input_0"));
  EXPECT_FALSE(append_binding_name(names, "other_0"));
}

} // namespace
} // namespace detail
} // namespace executorch_backend
} // namespace torch_tensorrt
