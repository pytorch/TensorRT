#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(LoweringPasses, EliminateExceptionOrPassPattern_Block0) {
  // parseIR does not support " = prim::If(%51)" with no return value
  /*std::string source_ir = R"IR(graph(%x.1 : Tensor, %y.1 : Tensor):
  %3 : NoneType = prim::Constant()
  %4 : int = prim::Constant[value=0]()
  %mod_list.1 : Tensor[] = prim::ListConstruct(%x.1)
  %47 : Tensor = aten::sum(%x.1, %3)
  %49 : Tensor = aten::sum(%y.1, %3)
  %50 : Tensor = aten::gt(%47, %49)
  %51 : bool = aten::Bool(%50)
   = prim::If(%51)
    block0():
      = prim::RaiseException(%45)
      -> ()
    block1():
      -> ()
  %z.1 : Tensor = aten::cat(%mod_list.1, %4)
  return (%z.1))IR";*/

  auto g = std::make_shared<torch::jit::Graph>();
  auto x = g->insertInput(0, "x");
  auto y = g->insertInput(1, "y");
  torch::jit::IValue zero(0);
  auto zero_const_val = g->insertConstant(zero);
  auto none_const_val = g->insertConstant(torch::jit::IValue());
  torch::jit::IValue except("EXCEPTION");
  auto except_val = g->insertConstant(except);
  auto list_node = g->createList(x->type(), torch::jit::ArrayRef<torch::jit::Value*>(x));
  g->insertNode(list_node);
  auto sum_x_node = g->create(torch::jit::aten::sum, {x, none_const_val});
  g->insertNode(sum_x_node);
  auto sum_y_node = g->create(torch::jit::aten::sum, {y, none_const_val});
  g->insertNode(sum_y_node);
  auto gt_node = g->create(torch::jit::aten::gt, {sum_x_node->output(), sum_y_node->output()});
  g->insertNode(gt_node);
  auto bool_node = g->create(torch::jit::aten::Bool, {gt_node->output()});
  bool_node->output()->setType(torch::jit::BoolType::get());
  g->insertNode(bool_node);
  auto if_node = g->create(torch::jit::prim::If, {bool_node->output()}, 0);
  auto if_block0 = if_node->addBlock();
  auto exception_node = g->create(torch::jit::prim::RaiseException, {except_val, none_const_val}, 0);
  if_block0->appendNode(exception_node);
  /*auto if_block1 =*/if_node->addBlock();
  g->insertNode(if_node);
  auto cat_node = g->create(torch::jit::aten::cat, {list_node->output(), zero_const_val});
  g->insertNode(cat_node);
  g->registerOutput(cat_node->output());

  std::cout << "Source Graph: " << *g << std::endl;
  torch_tensorrt::core::lowering::passes::EliminateExceptionOrPassPattern(g);
  std::cout << "Modified Graph: " << *g << std::endl;
  for (auto node : g->nodes()) {
    EXPECT_NE(node, if_node);
  }
}

TEST(LoweringPasses, EliminateExceptionOrPassPattern_Block1) {
  // parseIR does not support " = prim::If(%51)" with no return value
  /*std::string source_ir = R"IR(graph(%x.1 : Tensor, %y.1 : Tensor):
  %3 : NoneType = prim::Constant()
  %4 : int = prim::Constant[value=0]()
  %mod_list.1 : Tensor[] = prim::ListConstruct(%x.1)
  %47 : Tensor = aten::sum(%x.1, %3)
  %49 : Tensor = aten::sum(%y.1, %3)
  %50 : Tensor = aten::gt(%47, %49)
  %51 : bool = aten::Bool(%50)
   = prim::If(%51)
    block0():
      -> ()
    block1():
      = prim::RaiseException(%45)
      -> ()
  %z.1 : Tensor = aten::cat(%mod_list.1, %4)
  return (%z.1))IR";*/

  auto g = std::make_shared<torch::jit::Graph>();
  auto x = g->insertInput(0, "x");
  auto y = g->insertInput(1, "y");
  torch::jit::IValue zero(0);
  auto zero_const_val = g->insertConstant(zero);
  auto none_const_val = g->insertConstant(torch::jit::IValue());
  torch::jit::IValue except("EXCEPTION");
  auto except_val = g->insertConstant(except);
  auto list_node = g->createList(x->type(), torch::jit::ArrayRef<torch::jit::Value*>(x));
  g->insertNode(list_node);
  auto sum_x_node = g->create(torch::jit::aten::sum, {x, none_const_val});
  g->insertNode(sum_x_node);
  auto sum_y_node = g->create(torch::jit::aten::sum, {y, none_const_val});
  g->insertNode(sum_y_node);
  auto gt_node = g->create(torch::jit::aten::gt, {sum_x_node->output(), sum_y_node->output()});
  g->insertNode(gt_node);
  auto bool_node = g->create(torch::jit::aten::Bool, {gt_node->output()});
  bool_node->output()->setType(torch::jit::BoolType::get());
  g->insertNode(bool_node);
  auto if_node = g->create(torch::jit::prim::If, {bool_node->output()}, 0);
  /*auto if_block0 = */ if_node->addBlock();
  auto if_block1 = if_node->addBlock();
  auto exception_node = g->create(torch::jit::prim::RaiseException, {except_val, none_const_val}, 0);
  if_block1->appendNode(exception_node);
  g->insertNode(if_node);
  auto cat_node = g->create(torch::jit::aten::cat, {list_node->output(), zero_const_val});
  g->insertNode(cat_node);
  g->registerOutput(cat_node->output());

  std::cout << "Source Graph: " << *g << std::endl;
  torch_tensorrt::core::lowering::passes::EliminateExceptionOrPassPattern(g);
  std::cout << "Modified Graph: " << *g << std::endl;
  for (auto node : g->nodes()) {
    EXPECT_NE(node, if_node);
  }
}

TEST(LoweringPasses, EliminateExceptionOrPassPattern_Negative) {
  // parseIR does not support " = prim::If(%51)" with no return value
  /*std::string source_ir = R"IR(graph(%x.1 : Tensor, %y.1 : Tensor):
  %3 : NoneType = prim::Constant()
  %4 : int = prim::Constant[value=0]()
  %mod_list.1 : Tensor[] = prim::ListConstruct(%x.1)
  %47 : Tensor = aten::sum(%x.1, %3)
  %49 : Tensor = aten::sum(%y.1, %3)
  %50 : Tensor = aten::gt(%47, %49)
  %51 : bool = aten::Bool(%50)
   = prim::If(%51)
    block0():
      %10 : Tensor[] = aten::append(%mod_list.1, %y.1)
      -> ()
    block1():
      -> ()
  %z.1 : Tensor = aten::cat(%mod_list.1, %4)
  return (%z.1))IR";*/

  auto g = std::make_shared<torch::jit::Graph>();
  auto x = g->insertInput(0, "x");
  auto y = g->insertInput(1, "y");
  torch::jit::IValue zero(0);
  auto zero_const_val = g->insertConstant(zero);
  auto none_const_val = g->insertConstant(torch::jit::IValue());
  auto list_node = g->createList(x->type(), torch::jit::ArrayRef<torch::jit::Value*>(x));
  g->insertNode(list_node);
  auto sum_x_node = g->create(torch::jit::aten::sum, {x, none_const_val});
  g->insertNode(sum_x_node);
  auto sum_y_node = g->create(torch::jit::aten::sum, {y, none_const_val});
  g->insertNode(sum_y_node);
  auto gt_node = g->create(torch::jit::aten::gt, {sum_x_node->output(), sum_y_node->output()});
  g->insertNode(gt_node);
  auto bool_node = g->create(torch::jit::aten::Bool, {gt_node->output()});
  bool_node->output()->setType(torch::jit::BoolType::get());
  g->insertNode(bool_node);
  auto if_node = g->create(torch::jit::prim::If, {bool_node->output()}, 0);
  auto if_block0 = if_node->addBlock();
  auto append_node = g->create(torch::jit::aten::append, {list_node->output(), y});
  if_block0->appendNode(append_node);
  /*auto if_block1 = */ if_node->addBlock();
  g->insertNode(if_node);
  auto cat_node = g->create(torch::jit::aten::cat, {list_node->output(), zero_const_val});
  g->insertNode(cat_node);
  g->registerOutput(cat_node->output());

  torch_tensorrt::core::lowering::passes::EliminateExceptionOrPassPattern(g);
  int if_count = 0;
  for (auto node : g->nodes()) {
    if (node == if_node) {
      if_count++;
    }
  }
  EXPECT_EQ(1, if_count);
}
