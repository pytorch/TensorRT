#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"

TEST(Partitioning, GetDependencyNodes) {
  auto g = std::make_shared<torch::jit::Graph>();
  auto x = g->insertInput(0, "x");
  torch::jit::IValue key("THE_KEY");
  auto keyVal = g->insertConstant(key);
  auto dictNode = g->createDict(
      keyVal->type(),
      x->type(),
      torch::jit::ArrayRef<torch::jit::Value*>(),
      torch::jit::ArrayRef<torch::jit::Value*>());
  g->insertNode(dictNode);
  auto setNode = g->create(
      torch::jit::Symbol::fromQualString("aten::_set_item"),
      torch::jit::ArrayRef<torch::jit::Value*>{dictNode->output(), keyVal, x},
      0);
  g->insertNode(setNode);
  auto getNode = g->create(
      torch::jit::Symbol::fromQualString("aten::__getitem__"),
      torch::jit::ArrayRef<torch::jit::Value*>{dictNode->output(), keyVal},
      1);
  g->insertNode(getNode);
  g->registerOutput(getNode->output());
  g->dump();
  std::vector<torch::jit::Value*> values{getNode->output()};
  auto depNodes = torch_tensorrt::core::partitioning::getDependencyNodes(values);
  for (auto node : depNodes) {
    std::cout << *node << std::endl;
  }
  EXPECT_EQ(3UL, depNodes.size());
}
